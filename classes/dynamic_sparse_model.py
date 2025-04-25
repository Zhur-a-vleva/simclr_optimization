import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet50
from tqdm import tqdm

from linear_evaluation.linear_classification import LinearClassification
from metrics.metrics import Metrics
from classes.simclr import SimCLR


class DynamicSparse:
    def __init__(self, temperature, device, lr, epochs, dataset, logger,
                 sparsity=0.8, reallocation_interval=500, prune_rate=0.3):
        super().__init__()
        self.name = "dynamic_sparse"
        self.t = temperature
        self.device = device
        self.lr = lr
        self.epochs = epochs

        # Initialize model
        self.model = SimCLR(resnet50, output_dim=128).to(device)
        self.best_model = self.model

        # Dataset and evaluation objects
        self.dataset = dataset
        self.linear_classification = None
        self.metrics = Metrics(self)
        self.handle = self.metrics.start_gpu_monitoring()
        self.logger = logger

        # Dynamic sparse parameters
        self.sparsity = sparsity  # Target global sparsity
        self.reallocation_interval = reallocation_interval  # How often to reallocate parameters
        self.prune_rate = prune_rate  # Percentage of remaining weights to prune each time

        # Initialize sparse masks for all applicable parameter tensors
        self.masks = {}
        self.initialize_sparse_masks()

        # Set up optimizer
        self.optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)

        # Set up scheduler
        train_loader, _, _ = dataset.get_loaders()
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=epochs * len(train_loader), eta_min=1e-3
        )

        # Initialize step counter for parameter reallocation
        self.steps = 0

    def initialize_sparse_masks(self):
        """Initialize sparse masks for convolutional layers"""
        for name, param in self.model.named_parameters():
            # Skip the first layer, normalization parameters, and biases
            if (len(param.shape) <= 1 or  # Skip biases and BN params
                    "encoder.conv1" in name or  # Skip first layer
                    "encoder.fc" in name or  # Skip FC layer
                    "encoder.bn" in name):  # Skip BN layers
                continue

            # Create a binary mask initialized with 1s where weights are kept
            mask = torch.zeros_like(param, dtype=torch.bool, device=self.device)

            # Randomly select indices to keep based on sparsity
            n_weights = param.numel()
            n_keep = int(n_weights * (1 - self.sparsity))

            # Flatten, set random elements to 1, and reshape back
            flat_mask = mask.view(-1)
            indices = torch.randperm(n_weights, device=self.device)[:n_keep]
            flat_mask[indices] = True

            # Store the mask
            self.masks[name] = mask

            # Apply the mask to the parameter
            param.data = param.data * mask

    def apply_masks(self):
        """Apply all masks to their corresponding parameters"""
        for name, param in self.model.named_parameters():
            if name in self.masks:
                param.data = param.data * self.masks[name]

    def reallocate_parameters(self):
        """Dynamic sparse parameter reallocation based on magnitude and gradients"""
        # Step 1: Count total parameters for reallocation
        total_params = 0
        total_zeros = 0
        param_values = {}

        for name, param in self.model.named_parameters():
            if name in self.masks:
                # Count non-zero parameters
                non_zeros = self.masks[name].sum().item()
                total_params += non_zeros
                total_zeros += param.numel() - non_zeros

                # Store parameter values for later use
                param_values[name] = param.data.clone()

        # Calculate number of parameters to prune and regrow
        to_prune = int(total_params * self.prune_rate)

        # Step 2: Identify parameters to prune (smallest magnitude)
        all_values = []
        for name, param in self.model.named_parameters():
            if name in self.masks:
                # Get non-zero parameter values
                values = param.data[self.masks[name]]
                values_with_meta = [(name, i, v.abs().item())
                                    for i, v in enumerate(values)]
                all_values.extend(values_with_meta)

        # Sort by magnitude and select smallest
        all_values.sort(key=lambda x: x[2])
        to_prune_values = all_values[:to_prune]

        # Step 3: Prune selected parameters
        indices_to_prune = {}
        for name, idx, _ in to_prune_values:
            if name not in indices_to_prune:
                indices_to_prune[name] = []
            indices_to_prune[name].append(idx)

        for name, indices in indices_to_prune.items():
            # Convert flat indices to mask indices
            flat_mask = self.masks[name].view(-1)
            non_zero_indices = flat_mask.nonzero().view(-1)

            # Prune selected indices
            for idx in indices:
                flat_mask[non_zero_indices[idx]] = False

            # Apply updated mask
            self.masks[name] = flat_mask.view_as(self.masks[name])

        # Step 4: Calculate layer scores for parameter growth
        # Heuristic: layers with larger fractions of non-zero weights get more parameters
        layer_scores = {}
        for name, mask in self.masks.items():
            non_zero_ratio = mask.sum().float() / mask.numel()
            layer_scores[name] = non_zero_ratio

        # Normalize scores
        total_score = sum(layer_scores.values())
        if total_score > 0:
            layer_scores = {k: v / total_score for k, v in layer_scores.items()}

        # Step 5: Redistribute pruned parameters
        for name, score in layer_scores.items():
            # Calculate how many parameters to add to this layer
            to_grow = int(to_prune * score)

            # Find zero positions in the mask
            flat_mask = self.masks[name].view(-1)
            zero_indices = (~flat_mask).nonzero().view(-1)

            if len(zero_indices) > 0 and to_grow > 0:
                # Select random positions to grow
                n_to_grow = min(to_grow, len(zero_indices))
                grow_indices = zero_indices[torch.randperm(len(zero_indices))[:n_to_grow]]

                # Update mask
                flat_mask[grow_indices] = True
                self.masks[name] = flat_mask.view_as(self.masks[name])

        # Step 6: Apply updated masks to parameters
        self.apply_masks()

    def nt_xent_loss(self, z_i, z_j):
        """NT-Xent loss for contrastive learning"""
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
        """Validate the model on validation set"""
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
        """Train the model with dynamic sparse reparameterization"""
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

                # Apply masks before optimizer step to ensure zero weights stay zero
                self.apply_masks()

                self.optimizer.step()
                self.scheduler.step()

                # Ensure sparsity is maintained
                self.apply_masks()

                # Check if it's time to reallocate parameters
                self.steps += 1
                if self.steps % self.reallocation_interval == 0:
                    self.reallocate_parameters()

                epoch_loss += loss.item()

            # save metrics
            training_time = time.time() - start_time
            self.metrics.metrics["contrastive_loss"].append(epoch_loss / len(train_loader))
            self.metrics.metrics["training_time_per_epoch_sec"].append(training_time)
            self.metrics.metrics["gpu_utilization_percent"].append(self.metrics.gpu_utilization(self.handle))
            self.metrics.metrics["memory_usage_MB"].append(self.metrics.memory_usage())
            self.metrics.metrics["model_size_MB"].append(self.metrics.model_size())

            # Validation
            val_loss = self.validate(val_loader)
            self.logger.info(
                f"Epoch {epoch + 1}: Training Loss: {epoch_loss / len(train_loader)}, "
                f"Validation Loss: {val_loss}, "
                f"GPU Util: {self.metrics.metrics['gpu_utilization_percent'][-1]}%, "
                f"Training time per epoch: {self.metrics.metrics['training_time_per_epoch_sec'][-1]} sec,"
                f"Memory usage: {self.metrics.metrics['memory_usage_MB'][-1]} MB,"
                f"Model size: {self.metrics.metrics['model_size_MB'][-1]} MB"
            )

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
        """Load the best model for evaluation"""
        self.best_model.load_state_dict(torch.load(f"models/{self.name}"))
        self.linear_classification = LinearClassification(self)
        self.metrics.set_linear_classification(self.linear_classification)
