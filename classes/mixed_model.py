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


class PrunedDCL():
    def __init__(self, temperature, device, lr, epochs, dataset, logger, pruning_stages):
        super().__init__()
        self.name = "pruned_dcl"
        self.t = temperature
        self.device = device
        self.lr = lr
        self.epochs = epochs

        # Модель на базе реснета
        self.model = SimCLR(resnet50, output_dim=128).to(device)
        self.original_model = self.clone_model()  # для подсчёта FLOPs (как в исходном коде)
        self.best_model = self.model

        self.dataset = dataset
        self.linear_classification = None
        self.metrics = Metrics(self)
        self.handle = self.metrics.start_gpu_monitoring()
        self.logger = logger

        # Параметры для этапов pruning
        self.pruning_stages = pruning_stages
        self.current_stage = 0
        self.unit_registry = []  # реестр bottleneck-блоков для MI-анализа

        # Регистрируем все Bottleneck-блоки из ResNet50
        for name, module in self.model.encoder.named_modules():
            if isinstance(module, Bottleneck):
                self.unit_registry.append({
                    'name': name,
                    'module': module,
                    'type': 'bottleneck'
                })

        # Оптимизатор и планировщик
        self.optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
        train_loader, _, _ = dataset.get_loaders()
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=epochs * len(train_loader),
            eta_min=1e-3
        )

    def clone_model(self):
        """Клонируем модель, чтобы прикинуть FLOPs."""
        clone = SimCLR(resnet50, output_dim=128).to(self.device)
        clone.load_state_dict(self.model.state_dict())
        return clone

    # =============== DCL LOSS (вместо InfoNCE) ===================
    def dcl_loss(self, z_i, z_j):
        """
        Реализация Decoupled Contrastive Learning (DCL), убирая позитивный член из знаменателя,
        чтобы не было 'negative-positive coupling'. См. Yeh et al. (2022).
        """
        N = z_i.size(0)
        # Склеиваем в один батч, итоговый размер [2N, dim]
        z = torch.cat([z_i, z_j], dim=0)
        z = nn.functional.normalize(z, dim=1)

        # Матрица сходств [2N x 2N]
        sim = torch.matmul(z, z.T) / self.t
        exp_sim = torch.exp(sim)

        # Для i < N -> i+N, для i >= N -> i-N (положительные пары)
        pos_index = torch.arange(2 * N, device=self.device)
        pos_index[:N] += N
        pos_index[N:] -= N

        # Убираем из знаменателя диагональ (self-сходство) и позитив
        exp_sim.fill_diagonal_(0.)
        exp_sim[torch.arange(2 * N), pos_index] = 0.

        # Знаменатель = сумма по строке
        denom = exp_sim.sum(dim=1)  # [2N]
        # Числитель = e^( sim(i, pos_i) )
        pos_vals = torch.exp(sim[torch.arange(2 * N), pos_index])

        # loss = - log( pos / denom )
        loss = -torch.log(pos_vals / denom)
        return loss.mean()

    # ===============================

    # =========== MI-методы (как в Pruned) =========================
    def gaussian_mi_estimate(self, features, labels):
        """Оценка I(X;Y) через гауссову смесь (примерно)."""
        n_samples, n_features = features.shape
        class_probs = torch.bincount(labels) / n_samples

        overall_entropy = self.gaussian_entropy(features)
        conditional_entropy = 0.0
        for c in torch.unique(labels):
            class_mask = (labels == c)
            class_features = features[class_mask]
            conditional_entropy += class_probs[c] * self.gaussian_entropy(class_features)

        return (overall_entropy - conditional_entropy).item()

    def gaussian_entropy(self, X):
        """Энтропия на основе RBF kernel."""
        n_samples, n_features = X.shape
        pairwise_dists = torch.cdist(X, X) ** 2

        # sigma по медиане
        sigma = torch.median(pairwise_dists) / (2 * np.log(n_samples + 1))
        kernel_matrix = torch.exp(-pairwise_dists / (2 * sigma))

        entropy_estimate = -torch.log(torch.mean(kernel_matrix, dim=1)).mean()
        return entropy_estimate

    def estimate_units_mi(self, dataloader):
        """Собираем активации (output) в bottleneck-блоках, считаем MI."""
        self.model.eval()
        activations = {}
        labels_list = []

        hooks = []
        # Регистрируем хуки на каждый block
        for unit in self.unit_registry:
            module = unit['module']
            name = unit['name']

            def hook_fn(m, i, o, nm=name):
                activations[nm] = o.detach().flatten(start_dim=1)

            h = module.register_forward_hook(hook_fn)
            hooks.append(h)

        # Прогоняем через модель
        with torch.no_grad():
            for (x_i, _), y in dataloader:
                x_i = x_i.to(self.device)
                _ = self.model(x_i)  # forward
                labels_list.append(y.to(self.device))

        # Сняли хуки
        for h in hooks:
            h.remove()

        labels = torch.cat(labels_list)
        mi_values = []

        # Считаем MI для каждого Bottleneck
        for unit in self.unit_registry:
            name = unit['name']
            features = activations[name]
            mi = self.gaussian_mi_estimate(features, labels)
            mi_values.append(mi)

        return np.array(mi_values)

    def adaptive_clustering(self, mi_values):
        """Адаптивная кластеризация и выбор 'лучших' блоков."""
        if self.current_stage < self.pruning_stages // 2:
            n_clusters = max(2, len(mi_values) // 3)  # агрессивная обрезка
        else:
            n_clusters = max(2, len(mi_values) // 5)  # консервативная

        kmeans = KMeans(n_clusters=n_clusters)
        arr_reshaped = mi_values.reshape(-1, 1)
        clusters = kmeans.fit_predict(arr_reshaped)

        keep_indices = []
        for cluster_id in np.unique(clusters):
            cluster_mi = mi_values[clusters == cluster_id]
            # оставляем наиболее информативный (max MI)
            idx = np.argmax(cluster_mi)
            # надо понять, какой именно это индекс среди всего массива
            # np.where(...) вернуть настоящие индексы
            actual_idx = np.where(clusters == cluster_id)[0][idx]  # внутри кластера
            keep_indices.append(actual_idx)

        return keep_indices

    def prune_units(self, keep_indices):
        """
        Удаляем из layer4 (для примера) Bottleneck'и, которые не в keep_indices.
        По аналогии с исходным кодом.
        """
        all_layers = []
        for name, module in self.model.encoder.named_children():
            if name.startswith('layer'):
                all_layers.append(module)

        # layer4 - последний, индекса 3
        layer4_modules = []
        for idx, module in enumerate(all_layers[3]):
            if idx in keep_indices:
                layer4_modules.append(module)

        all_layers[3] = nn.Sequential(*layer4_modules)
        self.model.encoder = nn.Sequential(*all_layers)
        self.update_model_metrics()

    def update_model_metrics(self):
        """Обновить метрики params/flops."""
        params = sum(p.numel() for p in self.model.parameters())
        input_data = torch.randn(1, 3, 224, 224).to(self.device)
        flops, _ = profile(self.model, inputs=(input_data,))
        self.metrics.metrics['params'] = params
        self.metrics.metrics['flops'] = flops

    # =============================================================

    def validate(self, val_loader):
        self.model.eval()
        val_loss = 0
        with torch.no_grad():
            for (x_i, x_j), _ in val_loader:
                x_i, x_j = x_i.to(self.device), x_j.to(self.device)
                _, z_i = self.model(x_i)
                _, z_j = self.model(x_j)
                # Используем DCL-loss
                loss = self.dcl_loss(z_i, z_j)
                val_loss += loss.item()
        return val_loss / len(val_loader)

    def train(self):
        train_loader, val_loader, _ = self.dataset.get_loaders()

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
                x_i, x_j = x_i.to(self.device), x_j.to(self.device)
                self.optimizer.zero_grad()

                # forward pass
                _, z_i = self.model(x_i)
                _, z_j = self.model(x_j)

                # === DCL-loss ===
                loss = self.dcl_loss(z_i, z_j)
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

                epoch_loss += loss.item()

            training_time = time.time() - start_time
            self.metrics.metrics["contrastive_loss"].append(epoch_loss / len(train_loader))
            self.metrics.metrics["training_time_per_epoch_sec"].append(training_time)
            self.metrics.metrics["gpu_utilization_percent"].append(self.metrics.gpu_utilization(self.handle))
            self.metrics.metrics["memory_usage_MB"].append(self.metrics.memory_usage())
            self.metrics.metrics["model_size_MB"].append(self.metrics.model_size())

            # Этап pruning каждые 10 эпох, например
            if epoch % 10 == 0 and epoch > 0:
                mi_values = self.estimate_units_mi(train_loader)
                keep_indices = self.adaptive_clustering(mi_values)
                self.prune_units(keep_indices)
                self.logger.info(
                    f"Stage {self.current_stage}: "
                    f"Kept {len(keep_indices)} bottlenecks, "
                    f"Params: {self.metrics.metrics['params'] / 1e6:.2f}M, "
                    f"FLOPs: {self.metrics.metrics['flops'] / 1e9:.2f}G"
                )
                self.current_stage += 1

            # Валидация
            val_loss = self.validate(val_loader)
            self.logger.info(
                f"Epoch {epoch + 1}: Training Loss: {epoch_loss / len(train_loader):.4f}, "
                f"Validation Loss: {val_loss:.4f}, "
                f"GPU Util: {self.metrics.metrics['gpu_utilization_percent'][-1]}%, "
                f"Training time: {training_time:.2f} sec, "
                f"Memory usage: {self.metrics.metrics['memory_usage_MB'][-1]} MB, "
                f"Model size: {self.metrics.metrics['model_size_MB'][-1]} MB"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                early_stopping_counter = 0
                torch.save(self.model.state_dict(), f"models/{self.name}")
                self.logger.info(f"Model saved at epoch {epoch + 1}")
            else:
                early_stopping_counter += 1

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


