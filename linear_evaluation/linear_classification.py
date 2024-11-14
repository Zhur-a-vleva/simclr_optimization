import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, normalized_mutual_info_score
from torch.utils.data import DataLoader
from tqdm import tqdm


class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.linear(x)


class LinearClassification:

    def __init__(self, model):
        super().__init__()
        self.test_model = model
        self.device = model.device
        self.dataset = model.dataset
        self.logger = model.logger
        self.metrics = model.metrics

    def get_features(self, loader, model):
        model.eval()
        features = []
        labels = []
        inference_start = time.time()
        with torch.no_grad():
            for x, y in tqdm(loader, desc="Extracting Features", leave=False):
                if isinstance(x, (tuple, list)):
                    x = x[0]
                x = x.to(self.device)
                h, _ = model(x)
                features.append(h.cpu().numpy())
                labels.append(y.numpy())
        inference_time = time.time() - inference_start
        return np.concatenate(features), np.concatenate(labels), inference_time

    def get_features_tensors(self):
        train, val, test = self.dataset.get_data()

        train_features_linear, train_labels_linear, _ = self.get_features(
            DataLoader(train, batch_size=self.dataset.batch_size, pin_memory=True), self.test_model)
        val_features_linear, val_labels_linear, _ = self.get_features(
            DataLoader(val, batch_size=self.dataset.batch_size, pin_memory=True), self.test_model)
        test_features_linear, test_labels_linear, _ = self.get_features(
            DataLoader(test, batch_size=self.dataset.batch_size, pin_memory=True), self.test_model)

        train_features_linear = torch.tensor(train_features_linear).to(self.device)
        train_labels_linear = torch.tensor(train_labels_linear).long().to(self.device)
        val_features_linear = torch.tensor(val_features_linear).to(self.device)
        val_labels_linear = torch.tensor(val_labels_linear).long().to(self.device)
        test_features_linear = torch.tensor(test_features_linear).to(self.device)
        test_labels_linear = torch.tensor(test_labels_linear).long().to(self.device)

        return train_features_linear, train_labels_linear, val_features_linear, val_labels_linear, test_features_linear, test_labels_linear

    def evaluate(self):
        self.metrics.compute_inference_time()
        self.logger.info(f"Inference time: {self.metrics.metrcis["inference_time_sec"][-1]} sec")

        # linear evaluation
        for param in self.test_model.encoder.parameters():
            param.requires_grad = False

        (train_features_linear, train_labels_linear,
         val_features_linear, val_labels_linear,
         test_features_linear, test_labels_linear) = self.get_features_tensors()

        input_dim = train_features_linear.shape[1]
        num_classes = len(torch.unique(train_labels_linear))
        clf = LogisticRegressionModel(input_dim, num_classes).to(self.device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(clf.parameters(), lr=0.001)

        # early stopping parameters
        early_stopping_patience = 10
        best_val_loss = float('inf')
        early_stopping_counter = 0

        # training logistic regression with early stopping
        clf.train()
        num_epochs = 100

        self.logger.info(f"Logistic Regression started")
        self.logger.info(f"Constants set: EPOCHS={num_epochs}")

        for epoch in tqdm(range(num_epochs), desc="Epochs", position=0):
            # training step
            optimizer.zero_grad()
            outputs = clf(train_features_linear)
            loss = criterion(outputs, train_labels_linear)
            loss.backward()
            optimizer.step()

            # validation step
            clf.eval()
            with torch.no_grad():
                val_outputs = clf(val_features_linear)
                val_loss = criterion(val_outputs, val_labels_linear).item()
                self.logger.info(f"Training loss: {loss},"
                                 f"Validation loss: {val_loss}")

                # early stopping check
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    early_stopping_counter = 0
                    # save the best model
                    torch.save(clf.state_dict(), f"linear_evaluation/logistic_model_{self.test_model.name}")
                    self.logger.info(f"Model saved at epoch {epoch + 1}")
                else:
                    early_stopping_counter += 1
                    if early_stopping_counter >= early_stopping_patience:
                        self.logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                        break

            clf.train()  # return to training mode

        self.logger.info("Training complete")

        # load the best model for evaluation
        clf.load_state_dict(torch.load(f"linear_evaluation/logistic_model_{self.test_model.name}"))

        # final evaluation on test set
        clf.eval()
        with torch.no_grad():
            test_outputs = clf(test_features_linear)
            _, pred = torch.max(test_outputs, 1)
            accuracy = accuracy_score(test_labels_linear.cpu(), pred.cpu())
            nmi = normalized_mutual_info_score(test_labels_linear.cpu(), pred.cpu())

        self.metrics.metrics["linear_evaluation_accuracy"].append(accuracy)
        self.metrics.metrics["nmi"].append(nmi)

        self.logger.info(f"Linear evaluation accuracy: {self.metrics.metrics["linear_evaluation_accuracy"][-1]:.4f},"
                         f"NMI: {self.metrics.metrics["nmi"][-1]:.4f}")

        tqdm.write(f"Linear Evaluation Accuracy: {accuracy:.4f}, NMI: {nmi:.4f}")
