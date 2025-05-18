import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet50
from tqdm import tqdm

from linear_evaluation.linear_classification import LinearClassification
from metrics.metrics import Metrics
from classes.simclr import SimCLR


class ScheduleFreeSimCLR:
    def __init__(self, temperature, device, lr, epochs, dataset, logger):
        super().__init__()
        self.name = "schedule_free_simclr"
        self.t = temperature
        self.device = device
        self.lr = lr
        self.epochs = epochs
        self.model = SimCLR(resnet50, output_dim=128).to(device)
        self.best_model = self.model
        self.dataset = dataset
        self.linear_classification = None
        self.metrics = Metrics(self)
        self.handle = self.metrics.start_gpu_monitoring()
        self.logger = logger

        # Schedule-Free AdamW параметры
        beta1 = 0.9
        beta2 = 0.999
        weight_decay = 1e-4
        eps = 1e-8
        warmup_steps = int(0.05 * epochs * len(dataset.get_loaders()[0]))  # 5% от общего числа шагов

        # Инициализация Schedule-Free AdamW
        self.optimizer = self._create_schedule_free_adamw(
            self.model.parameters(),
            lr=lr,
            betas=(beta1, beta2),
            weight_decay=weight_decay,
            eps=eps,
            warmup_steps=warmup_steps
        )

        # Нет необходимости в scheduler, так как Schedule-Free исключает необходимость в расписании

    def _create_schedule_free_adamw(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2,
                                    warmup_steps=0):
        """
        Создает оптимизатор Schedule-Free AdamW, который реализует подход из статьи
        "The Road Less Scheduled" для оптимизации без использования расписаний
        """

        class ScheduleFreeAdamW(optim.Optimizer):
            def __init__(self, params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                         warmup_steps=warmup_steps):
                if not 0.0 <= betas[0] < 1.0:
                    raise ValueError(f"Invalid beta1 parameter: {betas[0]}")
                if not 0.0 <= betas[1] < 1.0:
                    raise ValueError(f"Invalid beta2 parameter: {betas[1]}")
                if eps < 0.0:
                    raise ValueError(f"Invalid epsilon value: {eps}")

                defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, warmup_steps=warmup_steps)
                super(ScheduleFreeAdamW, self).__init__(params, defaults)

                # Инициализация глобального счетчика шагов
                self.state['step'] = 0

            def step(self, closure=None):
                """Выполняет один шаг оптимизации."""
                loss = None
                if closure is not None:
                    loss = closure()

                # Увеличение глобального счетчика шагов
                self.state['step'] += 1
                current_step = self.state['step']

                for group in self.param_groups:
                    lr = group['lr']
                    beta1, beta2 = group['betas']
                    eps = group['eps']
                    weight_decay = group['weight_decay']
                    warmup_steps = group['warmup_steps']

                    # Применение разогрева при необходимости
                    if warmup_steps > 0:
                        lr = lr * min(1.0, current_step / warmup_steps)

                    # Расчет коэффициента взвешенного среднего c_t
                    c_t = current_step / (current_step + 1)

                    for p in group['params']:
                        if p.grad is None:
                            continue

                        grad = p.grad.data

                        state = self.state[p]

                        # Инициализация состояния, если необходимо
                        if len(state) == 0:
                            state['z'] = torch.zeros_like(p.data)  # Базовая точка оптимизации
                            state['x'] = p.data.clone()  # Текущий итерат (взвешенное среднее)
                            state['v'] = torch.zeros_like(p.data)  # Оценка второго момента

                        # Извлечение буферов
                        z = state['z']
                        x = state['x']
                        v = state['v']

                        # Расчет y_t (точка оценки градиента) с использованием интерполяции
                        y = beta1 * z + (1 - beta1) * x

                        # Применение weight decay в точке оценки градиента
                        if weight_decay != 0:
                            grad = grad.add(y, alpha=weight_decay)

                        # Обновление оценки второго момента
                        v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                        # Коррекция смещения для второго момента
                        v_hat = v.div(1 - beta2 ** current_step)

                        # Обновление Adam
                        denom = v_hat.sqrt().add_(eps)
                        step_size = lr / (1 - beta1 ** current_step)  # Коррекция смещения для первого момента

                        # Обновление z (шаг Adam)
                        z.addcdiv_(grad, denom, value=-step_size)

                        # Обновление x (взвешенное среднее)
                        x.mul_(c_t).add_(z, alpha=1 - c_t)

                        # Копирование x обратно в параметр
                        p.data.copy_(x)

                return loss

        return ScheduleFreeAdamW(params, lr, betas, eps, weight_decay, warmup_steps)

    def nt_xent_loss(self, z_i, z_j):
        """
        Нормализованная кросс-энтропия для контрастивного обучения.
        """
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
        """
        Валидация модели.
        """
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
        """
        Обучение модели с Schedule-Free AdamW.
        """
        train_loader, val_loader, _ = self.dataset.get_loaders()

        # Параметры раннего останова
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
                epoch_loss += loss.item()

            # Сохранение метрик
            training_time = time.time() - start_time
            self.metrics.metrics["contrastive_loss"].append(epoch_loss / len(train_loader))
            self.metrics.metrics["training_time_per_epoch_sec"].append(training_time)
            self.metrics.metrics["gpu_utilization_percent"].append(self.metrics.gpu_utilization(self.handle))
            self.metrics.metrics["memory_usage_MB"].append(self.metrics.memory_usage())
            self.metrics.metrics["model_size_MB"].append(self.metrics.model_size())

            # Валидация
            val_loss = self.validate(val_loader)
            self.logger.info(
                f"Epoch {epoch + 1}: Training Loss: {epoch_loss / len(train_loader)}, "
                f"Validation Loss: {val_loss}, "
                f"GPU Util: {self.metrics.metrics['gpu_utilization_percent'][-1]}%, "
                f"Training time per epoch: {self.metrics.metrics['training_time_per_epoch_sec'][-1]} sec,"
                f"Memory usage: {self.metrics.metrics['memory_usage_MB'][-1]} MB,"
                f"Model size: {self.metrics.metrics['model_size_MB'][-1]} MB")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                early_stopping_counter = 0
                torch.save(self.model.state_dict(), f"models/{self.name}")
                self.logger.info(f"Model saved at epoch {epoch + 1}")
            else:
                early_stopping_counter += 1

            # Проверка условия раннего останова
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
        """
        Загрузка лучшей модели после обучения.
        """
        self.best_model.load_state_dict(torch.load(f"models/{self.name}"))
        self.linear_classification = LinearClassification(self)
        self.metrics.set_linear_classification(self.linear_classification)
