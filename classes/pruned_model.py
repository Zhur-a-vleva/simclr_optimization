import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet50
from tqdm import tqdm
import numpy as np

from linear_evaluation.linear_classification import LinearClassification
from metrics.metrics import Metrics
from classes.simclr import SimCLR


class Pruned:
    def __init__(self, temperature, device, lr, epochs, dataset, logger):
        super().__init__()
        self.name = "pruned"
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
        # keep track of feature maps and their information content
        self.feature_maps = {}
        self.mutual_info = {}
        self.active_units = set(range(len(self.model.encoder._modules)))

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

    @staticmethod
    def compute_mutual_information(feature_maps, labels):
        """
        Compute mutual information between feature maps and output labels
        using Gaussian Mixture Model approximation with improved numerical stability.
        """
        mutual_info = {}

        for unit_idx, features in feature_maps.items():
            # flatten features to vectors
            features = features.reshape(features.size(0), -1)

            # separate features by class
            class_features = {}
            for i in range(len(labels)):
                label = labels[i].item()
                if label not in class_features:
                    class_features[label] = []
                class_features[label].append(features[i])

            # calculate means for each class
            class_means = {}
            for label, feats in class_features.items():
                class_means[label] = torch.stack(feats).mean(dim=0)

            # estimate sigma for GMM - using a more robust approach
            # use the average pairwise distance as a basis for sigma
            total_dist = 0.0
            count = 0
            for i in range(min(100, len(features))):  # sample to avoid O(n²) complexity
                for j in range(i + 1, min(100, len(features))):
                    dist = torch.sum((features[i] - features[j]) ** 2).item()
                    total_dist += dist
                    count += 1

            # set sigma based on average distance (avoid division by zero)
            sigma = np.sqrt(total_dist / max(1, count)) + 1e-8

            # upper bound on mutual information
            n = len(labels)

            # calculate first term: H(T) with numerical stability
            h_t = 0
            for i in range(min(100, n)):  # sample to reduce computation
                for j in range(min(100, n)):
                    if i != j:  # Skip self-comparisons
                        dist_sq = torch.sum((features[i] - features[j]) ** 2).item()
                        # use a numerically stable approach
                        h_t += -dist_sq / (2 * sigma ** 2)  # log space calculation

            sample_size = min(100, n)
            h_t = h_t / (sample_size * (sample_size - 1)) if sample_size > 1 else 0

            # calculate second term: H(T|Y) with numerical stability
            h_t_given_y = 0
            total_weight = 0

            for label, feats in class_features.items():
                n_k = len(feats)
                if n_k <= 1:  # skip classes with only one sample
                    continue

                class_sum = 0
                feats_tensor = torch.stack(feats)

                # sample if there are too many features in this class
                sample_size_k = min(100, n_k)
                indices = torch.randperm(n_k)[:sample_size_k]
                feats_tensor = feats_tensor[indices]

                for i in range(sample_size_k):
                    for j in range(sample_size_k):
                        if i != j:  # skip self-comparisons
                            dist_sq = torch.sum((feats_tensor[i] - feats_tensor[j]) ** 2).item()
                            # use a numerically stable approach
                            class_sum += -dist_sq / (4 * sigma ** 2)  # log space calculation

                norm_class_sum = class_sum / (sample_size_k * (sample_size_k - 1)) if sample_size_k > 1 else 0
                h_t_given_y += (n_k / n) * norm_class_sum
                total_weight += n_k / n

            # adjust if we didn't process all classes
            if total_weight > 0:
                h_t_given_y = h_t_given_y / total_weight

            # mutual information: I(T;Y) = H(T) - H(T|Y)
            mutual_info[unit_idx] = h_t - h_t_given_y

        return mutual_info

    @staticmethod
    def cluster_mutual_information(mi_values, num_clusters=3):
        """
        Cluster units based on their mutual information values
        and select centroids from each cluster to keep
        """
        # convert MI values to array for clustering
        mi_items = list(mi_values.items())
        unit_indices = [item[0] for item in mi_items]
        mi_array = np.array([item[1] for item in mi_items])

        # simple clustering based on value ranges
        sorted_idx = np.argsort(mi_array)
        cluster_size = len(sorted_idx) // num_clusters

        clusters = []
        for i in range(num_clusters):
            start_idx = i * cluster_size
            end_idx = (i + 1) * cluster_size if i < num_clusters - 1 else len(sorted_idx)
            cluster_units = [unit_indices[sorted_idx[j]] for j in range(start_idx, end_idx)]
            clusters.append(cluster_units)

        # select centroids (units closest to input layer in each cluster)
        centroids = []
        for cluster in clusters:
            if len(cluster) > 0:
                centroids.append(min(cluster))

        return centroids, clusters

    def prune_units(self, keep_units):
        """
        Prune the network to keep only the specified units
        """
        self.active_units = set(keep_units)
        self.logger.info(f"Keeping units: {sorted(self.active_units)}")

        pruned_model = SimCLR(resnet50, output_dim=128).to(self.device)

        orig_state_dict = self.model.state_dict()
        pruned_state_dict = pruned_model.state_dict()

        for name, param in orig_state_dict.items():
            if name in pruned_state_dict:
                keep_param = True
                for unit_idx in range(4):  # ResNet has 4 main layer groups
                    if unit_idx not in self.active_units and f"encoder.layer{unit_idx}" in name:
                        keep_param = False
                        break

                if keep_param:
                    pruned_state_dict[name] = param

        pruned_model.load_state_dict(pruned_state_dict, strict=False)
        self.model = pruned_model
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9, weight_decay=1e-4)
        train_loader, _, _ = self.dataset.get_loaders()
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.epochs * len(train_loader),
            eta_min=1e-3
        )

    def collect_feature_maps(self, data_loader):
        """
        Collect feature maps from all units of the network
        """
        feature_maps = {}
        all_labels = []

        # define hooks to capture intermediate activations
        activations = {}

        def get_activation(name):
            def hook(output):
                activations[name] = output.detach()

            return hook

        # register hooks for each layer
        hooks = [self.model.encoder.layer1.register_forward_hook(get_activation(0)),
                 self.model.encoder.layer2.register_forward_hook(get_activation(1)),
                 self.model.encoder.layer3.register_forward_hook(get_activation(2)),
                 self.model.encoder.layer4.register_forward_hook(get_activation(3))]

        # for ResNet, we want to capture the output of each major layer block
        # these are the high-level layer groups in ResNet

        self.model.eval()
        with torch.no_grad():
            for i, ((x, _), y) in enumerate(data_loader):
                if i >= 10:  # limit the number of batches for efficiency
                    break

                x = x.to(self.device)
                y = y.to(self.device)

                _ = self.model.encoder(x)

                for unit_idx in activations:
                    if unit_idx not in feature_maps:
                        feature_maps[unit_idx] = []

                    # apply global average pooling to make feature dimension consistent
                    act = activations[unit_idx]
                    pooled_feature = nn.functional.adaptive_avg_pool2d(act, 1).squeeze(-1).squeeze(-1)
                    feature_maps[unit_idx].append(pooled_feature)

                all_labels.append(y)

        # remove hooks
        for hook in hooks:
            hook.remove()

        for unit_idx in feature_maps:
            feature_maps[unit_idx] = torch.cat(feature_maps[unit_idx], dim=0)

        all_labels = torch.cat(all_labels, dim=0)

        return feature_maps, all_labels

    def train(self, num_pruning_stages=3, pruning_interval=10):
        train_loader, val_loader, _ = self.dataset.get_loaders()

        # early stopping
        early_stopping_patience = 100
        best_val_loss = float('inf')
        early_stopping_counter = 0

        epoch_bar = tqdm(range(self.epochs), desc="Epochs", position=0)

        for epoch in epoch_bar:
            # check if it's time to prune
            if epoch > 0 and epoch % pruning_interval == 0 and len(self.active_units) > num_pruning_stages:
                # collect feature maps for pruning decision
                self.logger.info(f"Collecting feature maps for pruning at epoch {epoch + 1}")
                feature_maps, labels = self.collect_feature_maps(train_loader)

                # compute mutual information
                mi_values = self.compute_mutual_information(feature_maps, labels)
                self.logger.info(f"Mutual Information values: {mi_values}")

                # cluster and select units to keep
                keep_units, clusters = self.cluster_mutual_information(mi_values, num_clusters=3)

                self.prune_units(keep_units)
                self.logger.info(f"Pruned network at epoch {epoch + 1}, keeping {len(keep_units)} units")
                self.logger.info(f"Clusters: {clusters}")
                self.logger.info(
                    f"Network now has {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,} parameters")

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

            # validation
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
        self.best_model.load_state_dict(torch.load(f"models/{self.name}"))
        self.linear_classification = LinearClassification(self)
        self.metrics.set_linear_classification(self.linear_classification)
