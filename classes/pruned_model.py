import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.cluster import KMeans
from torchvision.models import resnet50
from tqdm import tqdm
from thop import profile
from torchvision.models.resnet import Bottleneck

from classes.simclr import SimCLR
from linear_evaluation.linear_classification import LinearClassification
from metrics.metrics import Metrics


class Pruned:
    def __init__(self, temperature, device, lr, epochs, dataset, logger, pruning_stages, max_samples_for_mi=1000):
        super().__init__()
        self.name = "pruned"
        self.t = temperature
        self.device = device
        self.lr = lr
        self.epochs = epochs
        self.model = SimCLR(resnet50, output_dim=128).to(device)
        self.original_model = self.clone_model()  # Для подсчёта FLOPs
        self.best_model = self.model
        self.dataset = dataset
        self.linear_classification = None
        self.metrics = Metrics(self)
        self.handle = self.metrics.start_gpu_monitoring()
        self.logger = logger
        self.pruning_stages = pruning_stages
        self.max_samples_for_mi = max_samples_for_mi
        self.current_stage = 0
        self.unit_registry = []  # Регистр блоков для MI

        # Initialize metrics immediately
        self.update_model_metrics()

        """Идентификация блоков в стандартном ResNet50"""
        for name, module in self.model.encoder.named_modules():
            # Регистрируем все Bottleneck блоки
            if isinstance(module, Bottleneck):
                self.unit_registry.append({
                    'name': name,
                    'module': module,
                    'type': 'bottleneck'
                })

        self.optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)

        train_loader, _, _ = dataset.get_loaders()
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs * len(train_loader),
                                                                    eta_min=1e-3)

    def clone_model(self):
        """Клонирование модели для подсчёта FLOPs"""
        clone = SimCLR(resnet50, output_dim=128).to(self.device)
        clone.load_state_dict(self.model.state_dict())
        return clone

    def gaussian_mi_estimate(self, features, labels):
        n_samples, n_features = features.shape
        class_probs = torch.bincount(labels) / n_samples

        overall_entropy = self.gaussian_entropy(features)
        conditional_entropy = 0.0

        for c in torch.unique(labels):
            class_mask = (labels == c)
            class_features = features[class_mask]
            class_entropy = self.gaussian_entropy(class_features)
            conditional_entropy += class_probs[c] * class_entropy

        return (overall_entropy - conditional_entropy).item()

    def gaussian_entropy(self, X, sample_size=500):  # Уменьшить sample_size для ускорения
        n_samples, n_features = X.shape

        # Используем подвыборку для вычисления sigma
        if n_samples > sample_size:
            indices = torch.randperm(n_samples)[:sample_size]
            X_sample = X[indices]
        else:
            X_sample = X

        pairwise_dists = torch.cdist(X_sample, X_sample) ** 2
        sigma = torch.median(pairwise_dists) / (2 * np.log(n_samples + 1))

        # Вычисляем kernel_matrix для всех точек (теперь n_samples <= 1000)
        kernel_matrix = torch.exp(-torch.cdist(X, X) ** 2 / (2 * sigma))
        entropy_estimate = -torch.log(kernel_matrix.mean(dim=1)).mean()

        return entropy_estimate

    def estimate_units_mi(self, dataloader):
        self.model.eval()
        mi_values = []

        for unit in self.unit_registry:
            activations = []
            labels = []
            collected = 0  # Счетчик собранных образцов
            hook = None

            # Регистрация хука только для текущего блока
            def save_activation(m, i, o):
                nonlocal collected
                features = o.detach().flatten(start_dim=1)
                activations.append(features.cuda())  # Хранить на GPU
                collected += features.shape[0]
                if collected >= self.max_samples_for_mi:
                    self.stop_data_collection = True  # Флаг для остановки сбора данных

            hook = unit['module'].register_forward_hook(save_activation)

            self.stop_data_collection = False

            with torch.no_grad():
                for (x_i, _), y in dataloader:
                    if self.stop_data_collection:
                        break
                    x_i = x_i.to(self.device)
                    self.model(x_i)
                    labels.append(y.to(self.device))

            # Объединение активаций и меток
            features = torch.cat(activations, dim=0)[:self.max_samples_for_mi]
            labels = torch.cat(labels)[:self.max_samples_for_mi].to(self.device)

            mi = self.gaussian_mi_estimate(features, labels)
            mi_values.append(mi)

            # Очистка памяти
            hook.remove()
            del activations, features, labels
            torch.cuda.empty_cache()  # Очистка GPU-памяти

        return np.array(mi_values)

    def adaptive_clustering(self, mi_values):
        """Адаптивная кластеризация на основе плотности MI"""
        if self.current_stage < self.pruning_stages // 2:
            n_clusters = max(2, len(mi_values) // 3)  # Агрессивная обрезка
        else:
            n_clusters = max(2, len(mi_values) // 5)  # Консервативная

        kmeans = KMeans(n_clusters=n_clusters)
        clusters = kmeans.fit_predict(mi_values.reshape(-1, 1))

        keep_indices = []
        for cluster_id in np.unique(clusters):
            cluster_mi = mi_values[clusters == cluster_id]
            keep_idx = np.argmax(cluster_mi)  # Сохраняем наиболее информативные
            keep_indices.append(keep_idx)

        return keep_indices

    def prune_units(self, keep_indices):
        """Обрезка блоков в ResNet50 с сохранением структуры"""
        all_layers = list(self.model.encoder.children())  # Собираем все слои модели

        # Проверяем, является ли layer4 Sequential (если нет, пропускаем обрезку)
        if not isinstance(all_layers[3], nn.Sequential):
            return  # layer4 уже обрезан до Identity, ничего не делаем

        # Получаем текущий layer4
        layer4 = all_layers[3]

        # Фильтруем блоки внутри layer4 по keep_indices
        layer4_modules = []
        for idx, module in enumerate(layer4):  # Теперь точно работаем с Sequential
            if idx in keep_indices:
                layer4_modules.append(module)

        # Если не осталось блоков, заменяем layer4 на Identity
        if not layer4_modules:
            all_layers[3] = nn.Identity()  # Заменяем на Identity, если нет блоков
        else:
            all_layers[3] = nn.Sequential(*layer4_modules)  # Обрезанный layer4

        # Пересобираем модель
        self.model.encoder = nn.Sequential(*all_layers)
        self.update_model_metrics()

    def update_model_metrics(self):
        """Обновление метрик модели"""
        # Подсчёт параметров и немедленное сохранение
        params = sum(p.numel() for p in self.model.parameters())
        self.metrics.metrics['params'] = params  # Сохраняем параметры немедленно

        # Подсчёт FLOPs с обработкой ошибок
        try:
            input = torch.randn(1, 3, 224, 224).to(self.device)
            flops, _ = profile(self.model, inputs=(input,))
            self.metrics.metrics['flops'] = flops
        except Exception as e:
            self.logger.error(f"Ошибка при расчете FLOPs: {e}")
            self.metrics.metrics['flops'] = None  # Можно установить значение по умолчанию или None

    def nt_xent_loss(self, z_i, z_j):
        N = 2 * z_i.size(0)
        z = torch.cat((z_i, z_j), dim=0)
        z = nn.functional.normalize(z, dim=1)
        similarity_matrix = torch.matmul(z, z.T) / self.t
        mask = (~torch.eye(N, N, dtype=bool)).to(self.device)
        exp_sim = torch.exp(similarity_matrix) * mask
        sum_exp_sim = exp_sim.sum(dim=1, keepdim=True)
        positive_sim = torch.exp(torch.sum(z_i * z_j, dim=1) / self.t)
        loss = -torch.log(positive_sim / sum_exp_sim[:z_i.size(0)])
        return loss.mean()

    def validate(self, val_loader):
        self.model.eval()
        val_loss = 0
        with torch.no_grad():
            for (x_i, x_j), _ in val_loader:
                x_i, x_j = x_i.to(self.device), x_j.to(self.device)
                _, z_i = self.model(x_i)
                _, z_j = self.model(x_j)
                loss = self.nt_xent_loss(z_i, z_j)
                val_loss += loss.item()

        return val_loss / len(val_loader)

    def train(self):
        train_loader, val_loader, _ = self.dataset.get_loaders()

        # Early stopping
        early_stopping_patience = 100
        best_val_loss = float('inf')
        early_stopping_counter = 0

        epoch_bar = tqdm(range(self.epochs), desc="Epochs", position=0)

        for epoch in epoch_bar:
            self.model.train()
            epoch_loss = 0
            start_time = time.time()

            batch_bar = tqdm(train_loader, desc=f"Training Epoch {epoch + 1}", leave=False, position=1)

            for (x_i, x_j), _ in batch_bar:
                x_i = x_i.to(self.device)
                x_j = x_j.to(self.device)

                self.optimizer.zero_grad()
                _, z_i = self.model(x_i)
                _, z_j = self.model(x_j)
                loss = self.nt_xent_loss(z_i, z_j)
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                epoch_loss += loss.item()

            # save metrics
            training_time = time.time() - start_time
            self.metrics.metrics["contrastive_loss"].append(epoch_loss / len(train_loader))
            self.metrics.metrics["training_time_per_epoch_sec"].append(training_time)
            self.metrics.metrics["gpu_utilization_percent"].append(self.metrics.gpu_utilization(self.handle))
            self.metrics.metrics["memory_usage_MB"].append(self.metrics.memory_usage())
            self.metrics.metrics["model_size_MB"].append(self.metrics.model_size())

            # Этап обрезки
            if epoch % 10 == 0 and epoch > 0:
                mi_values = self.estimate_units_mi(train_loader)
                keep_indices = self.adaptive_clustering(mi_values)
                self.prune_units(keep_indices)

                self.logger.info(
                    f"Stage {self.current_stage}: "
                    f"Kept {len(keep_indices)} units, "
                    f"Params: {self.metrics.metrics['params'] / 1e6:.2f}M, "
                    f"FLOPs: {self.metrics.metrics['flops'] / 1e9:.2f}G"
                )
                self.current_stage += 1

            # Валидация
            val_loss = self.validate(val_loader)
            self.logger.info(
                f"Epoch {epoch + 1}: Training Loss: {epoch_loss / len(train_loader)}, "
                f"Validation Loss: {val_loss}, "
                f"GPU Util: {self.metrics.metrics['gpu_utilization_percent'][-1]}%, "
                f"Training time per epoch: {self.metrics.metrics['training_time_per_epoch_sec'][-1]} sec,"
                f"Memory usage: {self.metrics.metrics['memory_usage_MB'][-1]} MB,"
                f"Model size: {self.metrics.metrics["model_size_MB"][-1]} MB")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                early_stopping_counter = 0
                torch.save(self.model.state_dict(), f"models/{self.name}")
                self.logger.info(f"Model saved at epoch {epoch + 1}")
            else:
                early_stopping_counter += 1

            # check early stopping condition
            if early_stopping_counter >= early_stopping_patience:
                self.logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                break

            epoch_bar.set_postfix({
                "Loss": f"{epoch_loss / len(train_loader):.4f}",
                "Val Loss": f"{val_loss:.4f}",
                "Time": f"{training_time:.2f}s",
                "GPU Util": f"{self.metrics.metrics['gpu_utilization_percent'][-1]}%"
            })
            tqdm.write(f"Loss: {epoch_loss / len(train_loader):.4f}")
            tqdm.write(f"Val Loss: {val_loss:.4f}")

        self.logger.info("Training complete")

    def load_best_model(self):
        self.best_model.load_state_dict(torch.load(f"models/{self.name}"))
        self.linear_classification = LinearClassification(self)
        self.metrics.set_linear_classification(self.linear_classification)
