import os

import psutil
import torch
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetUtilizationRates


class Metrics:
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.linear = None
        self.dataset = model.dataset
        self.metrics = {
            "linear_evaluation_accuracy": [],
            "contrastive_loss": [],
            "nmi": [],
            "memory_usage_MB": [],
            "model_size_MB": [],
            "inference_time_sec": [],
            "training_time_per_epoch_sec": [],
            "gpu_utilization_percent": []
        }

    def set_linear_classification(self, linear_classification):
        self.linear = linear_classification

    def start_gpu_monitoring(self):
        nvmlInit()
        handle = nvmlDeviceGetHandleByIndex(0)  # GPU with index 0
        return handle

    def gpu_utilization(self, handle):
        return nvmlDeviceGetUtilizationRates(handle).gpu

    # memory usage monitoring
    def memory_usage(self):
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 ** 2)  # return in MB

    # model size monitoring
    def model_size(self):
        torch.save(self.model.model.state_dict(), "temp.p")
        size = os.path.getsize("temp.p") / (1024 ** 2)  # return in MB
        os.remove("temp.p")
        return size

    def compute_inference_time(self):
        train_loader, val_loader, test_loader = self.dataset.get_loaders()
        _, _, train_inference_time = self.linear.get_features(train_loader, self.model)
        _, _, val_inference_time = self.linear.get_features(val_loader, self.model)
        _, _, test_inference_time = self.linear.get_features(test_loader, self.model)
        self.metrics["inference_time_sec"].append(train_inference_time + val_inference_time + test_inference_time)
