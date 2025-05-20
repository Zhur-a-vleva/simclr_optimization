import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet50
from tqdm import tqdm

from linear_evaluation.linear_classification import LinearClassification
from metrics.metrics import Metrics
from classes.simclr import SimCLR


class DCL:
    def __init__(self, temperature, device, lr, epochs, dataset, logger):
        super().__init__()
        self.name = "dcl"
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
        self.optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)

        train_loader, _, _ = dataset.get_loaders()
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs * len(train_loader),
                                                                    eta_min=1e-3)

    def dcl_loss(self, z_i, z_j):
        """
        Decoupled Contrastive Learning loss from the paper
        The key difference from nt_xent_loss is that positive pairs are
        removed from the denominator of the InfoNCE loss
        """
        N = z_i.size(0)
        z_i = nn.functional.normalize(z_i, dim=1)
        z_j = nn.functional.normalize(z_j, dim=1)

        # positive similarity
        positive_sim = torch.exp(torch.sum(z_i * z_j, dim=1) / self.t)

        # all similarity for negatives
        z = torch.cat((z_i, z_j), dim=0)
        similarity_matrix = torch.matmul(z_i, z.T) / self.t

        # create mask for negatives
        mask = torch.ones((N, 2 * N), dtype=bool, device=self.device)
        mask[:, :N] = torch.eye(N, dtype=bool, device=self.device)
        mask[range(N), range(N, 2 * N)] = False

        # get negative similarities and sum them
        neg_sim = torch.exp(similarity_matrix) * mask
        neg_sim_sum = neg_sim.sum(dim=1)

        loss = -torch.log(positive_sim / neg_sim_sum)
        return loss.mean()

    def dclw_loss(self, z_i, z_j, sigma=0.5):
        """
        Weighted Decoupled Contrastive Learning loss
        Adds a weighting function to emphasize harder positive pairs
        """
        N = z_i.size(0)
        z_i = nn.functional.normalize(z_i, dim=1)
        z_j = nn.functional.normalize(z_j, dim=1)

        # compute positive similarities
        pos_sim = torch.sum(z_i * z_j, dim=1)

        # create weights for positive pairs based on their similarity
        weights = torch.exp(-pos_sim / sigma)

        # all similarity for negatives
        z = torch.cat((z_i, z_j), dim=0)
        similarity_matrix = torch.matmul(z_i, z.T) / self.t

        # create mask for negatives
        mask = torch.ones((N, 2 * N), dtype=bool, device=self.device)
        mask[:, :N] = torch.eye(N, dtype=bool, device=self.device)
        mask[range(N), range(N, 2 * N)] = False

        # get negative similarities and sum them
        neg_sim = torch.exp(similarity_matrix) * mask
        neg_sim_sum = neg_sim.sum(dim=1)

        loss = -weights * torch.log(torch.exp(pos_sim / self.t) / neg_sim_sum)
        return loss.mean()

    def validate(self, val_loader):
        self.model.eval()
        val_loss = 0
        with torch.no_grad():
            for (x_i, x_j), _ in val_loader:
                x_i, x_j = x_i.to(self.device), x_j.to(self.device)
                _, z_i = self.model(x_i)
                _, z_j = self.model(x_j)
                loss = self.dcl_loss(z_i, z_j)
                val_loss += loss.item()

        return val_loss / len(val_loader)

    def train(self, use_weighted=False):
        train_loader, val_loader, _ = self.dataset.get_loaders()

        # early stopping
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

                if use_weighted:
                    loss = self.dclw_loss(z_i, z_j)
                else:
                    loss = self.dcl_loss(z_i, z_j)

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

            # validation
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


class DCLW(DCL):
    """
    Weighted Decoupled Contrastive Learning implementation
    Inherits from DCL but uses the weighted loss by default
    """

    def __init__(self, temperature, device, lr, epochs, dataset, logger, sigma=0.5):
        super().__init__(temperature, device, lr, epochs, dataset, logger)
        self.name = "dclw"
        self.sigma = sigma

    def train(self, **kwargs):
        super().train(use_weighted=True)
