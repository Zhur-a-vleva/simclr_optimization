import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import json

# create plots directory if it doesn't exist
os.makedirs('plots', exist_ok=True)

# define file paths and model names
file_paths = [
    os.path.join("..", 'metrics/metrics_baseline.json'),
    os.path.join("..", "metrics/metrics_dcl.json"),
    os.path.join("..", 'metrics/metrics_dynamic_sparse.json'),
    os.path.join("..", 'metrics/metrics_pruned.json'),
    os.path.join("..", 'metrics/metrics_schedule_free_simclr.json')
]
model_names = ['Baseline', 'DCL', 'Dynamic Sparse', 'Pruned', 'Schedule Free SimCLR']

# load metrics data
metrics_data = {}
for file_path, model_name in zip(file_paths, model_names):
    try:
        with open(file_path, 'r') as f:
            metrics_data[model_name] = json.load(f)
    except FileNotFoundError:
        print(f"Warning: {file_path} not found!")
        metrics_data[model_name] = {}

# set up a consistent color scheme
colors_diff = ['blue', 'green', 'red', 'purple', 'orange']
plt.style.use('seaborn-v0_8-darkgrid')

# linear evaluation accuracy
plt.figure(figsize=(10, 6))
final_accuracy = []
for model_name, data in metrics_data.items():
    if 'linear_evaluation_accuracy' in data and data['linear_evaluation_accuracy']:
        final_accuracy.append(data['linear_evaluation_accuracy'][-1])

max_index = final_accuracy.index(max(final_accuracy))
colors = ['blue'] * len(final_accuracy)
colors[max_index] = 'green'

sns.barplot(x=model_names, y=final_accuracy, palette=colors)
for index, value in enumerate(final_accuracy):
    plt.text(index, value, f'{value:.3f}', ha='center', va='bottom')
plt.xlabel('Model')
plt.ylabel('Linear Evaluation Accuracy')
plt.title('Final Linear Evaluation Accuracy')
plt.tight_layout()
plt.savefig('plots/final_accuracy.png', dpi=300)
plt.close()

# NMI
plt.figure(figsize=(10, 6))
final_nmi = []
for model_name, data in metrics_data.items():
    if 'nmi' in data and data['nmi']:
        final_nmi.append(data['nmi'][-1])

max_index = final_nmi.index(max(final_nmi))
colors = ['blue'] * len(final_nmi)
colors[max_index] = 'green'

sns.barplot(x=model_names, y=final_nmi, palette=colors)
for index, value in enumerate(final_nmi):
    plt.text(index, value, f'{value:.3f}', ha='center', va='bottom')
plt.xlabel('Model')
plt.ylabel('NMI')
plt.title('Final Normalized Mutual Information (NMI)')
plt.tight_layout()
plt.savefig('plots/final_nmi.png', dpi=300)
plt.close()

# inference time sec
plt.figure(figsize=(10, 6))
inference_time_sec = []
for model_name, data in metrics_data.items():
    if 'inference_time_sec' in data and data['inference_time_sec']:
        inference_time_sec.append(data['inference_time_sec'][-1])

max_index = inference_time_sec.index(min(inference_time_sec))
colors = ['blue'] * len(inference_time_sec)
colors[max_index] = 'green'

sns.barplot(x=model_names, y=inference_time_sec, palette=colors)
for index, value in enumerate(inference_time_sec):
    plt.text(index, value, f'{value:.1f}', ha='center', va='bottom')
plt.xlabel('Model')
plt.ylabel('Inference Time Sec')
plt.title('Inference time in seconds')
plt.tight_layout()
plt.savefig('plots/inference_time_sec.png', dpi=300)
plt.close()

# contrastive loss
plt.figure(figsize=(10, 6))
for i, (model_name, data) in enumerate(metrics_data.items()):
    if 'contrastive_loss' in data:
        plt.plot(data['contrastive_loss'], label=model_name, color=colors_diff[i])
plt.xlabel('Epoch')
plt.ylabel('Contrastive Loss')
plt.title('Training Loss Comparison')
plt.legend()
plt.tight_layout()
plt.savefig('plots/contrastive_loss.png', dpi=600)
plt.close()

# memory usage
plt.figure(figsize=(10, 6))
for i, (model_name, data) in enumerate(metrics_data.items()):
    if 'memory_usage_MB' in data:
        if model_name not in ["Schedule Free SimCLR"]:
            plt.plot(data['memory_usage_MB'], label=model_name, color=colors_diff[i])
plt.xlabel('Epoch')
plt.ylabel('Memory Usage (MB)')
plt.title('Memory Usage per Epoch (without Schedule Free SimCLR)')
plt.legend()
plt.tight_layout()
plt.savefig('plots/memory_usage.png', dpi=600)
plt.close()

plt.figure(figsize=(10, 6))
for i, (model_name, data) in enumerate(metrics_data.items()):
    if 'memory_usage_MB' in data:
        if model_name in ["Schedule Free SimCLR"]:
            plt.plot(data['memory_usage_MB'], label=model_name, color=colors_diff[i])
plt.xlabel('Epoch')
plt.ylabel('Memory Usage (MB)')
plt.title('Memory Usage per Epoch (Schedule Free SimCLR)')
plt.legend()
plt.tight_layout()
plt.savefig('plots/memory_usage_schedule_free_simclr.png', dpi=600)
plt.close()

# model size
plt.figure(figsize=(10, 6))
model_size = []
for model_name, data in metrics_data.items():
    if 'model_size_MB' in data and data['model_size_MB']:
        model_size.append(data['model_size_MB'][-1])

max_index = model_size.index(min(model_size))
colors = ['blue'] * len(model_size)
colors[max_index] = 'green'

sns.barplot(x=model_names, y=model_size, palette=colors)
for index, value in enumerate(model_size):
    plt.text(index, value, f'{value:.4f}', ha='center', va='bottom')
plt.xlabel('Model')
plt.ylabel('Model Size (MB)')
plt.title('Model Size in MB')
plt.tight_layout()
plt.savefig('plots/model_size.png', dpi=300)
plt.close()

# training time per epoch
cumulative_baseline = 0
cumulative_dcl = 0
cumulative_dynamic_sparse = 0
cumulative_pruned = 0
cumulative_schedule = 0

plt.figure(figsize=(10, 6))
for i, (model_name, data) in enumerate(metrics_data.items()):
    if 'training_time_per_epoch_sec' in data:
        if model_name == "Dynamic Sparse":
            cumulative_dynamic_sparse = sum(data['training_time_per_epoch_sec'])
            continue
        elif model_name == "Baseline":
            cumulative_baseline = sum(data['training_time_per_epoch_sec'])
        elif model_name == "DCL":
            cumulative_dcl = sum(data['training_time_per_epoch_sec'])
        elif model_name == "Schedule Free SimCLR":
            cumulative_schedule = sum(data['training_time_per_epoch_sec'])
            continue
        else:
            cumulative_pruned = sum(data['training_time_per_epoch_sec'])
        plt.plot(data['training_time_per_epoch_sec'], label=model_name, color=colors_diff[i])
plt.xlabel('Epoch')
plt.ylabel('Training Time (seconds)')
plt.title('Training Time per Epoch (without Dynamic Sparse and Schedule Free SimCLR)')
plt.legend()
plt.tight_layout()
plt.savefig('plots/training_time.png', dpi=600)
plt.close()

plt.figure(figsize=(10, 6))
for i, (model_name, data) in enumerate(metrics_data.items()):
    if 'training_time_per_epoch_sec' in data:
        if model_name not in ["Dynamic Sparse"]:
            continue
        plt.plot(data['training_time_per_epoch_sec'], label=model_name, color=colors_diff[i])
plt.xlabel('Epoch')
plt.ylabel('Training Time (seconds)')
plt.title('Training Time per Epoch (Dynamic Sparse)')
plt.legend()
plt.tight_layout()
plt.savefig('plots/training_time_dynamic_sparse.png', dpi=600)
plt.close()

plt.figure(figsize=(10, 6))
for i, (model_name, data) in enumerate(metrics_data.items()):
    if 'training_time_per_epoch_sec' in data:
        if model_name not in ["Schedule Free SimCLR"]:
            continue
        plt.plot(data['training_time_per_epoch_sec'], label=model_name, color=colors_diff[i])
plt.xlabel('Epoch')
plt.ylabel('Training Time (seconds)')
plt.title('Training Time per Epoch (Schedule Free SimCLR)')
plt.legend()
plt.tight_layout()
plt.savefig('plots/training_time_schedule_free_simclr.png', dpi=600)
plt.close()

plt.figure(figsize=(10, 6))
cumulative = [cumulative_baseline / 3600.0, cumulative_dcl / 3600.0, cumulative_dynamic_sparse / 3600.0, cumulative_pruned / 3600.0, cumulative_schedule / 3600.0]

max_index = cumulative.index(min(cumulative))
colors = ['blue'] * len(cumulative)
colors[max_index] = 'green'

sns.barplot(x=model_names, y=cumulative, palette=colors)
for index, value in enumerate(cumulative):
    plt.text(index, value, f'{value:.2f}', ha='center', va='bottom')
plt.xlabel('Model')
plt.ylabel('Training Time Cumulative, hours')
plt.title('Training time in hours')
plt.tight_layout()
plt.savefig('plots/training_time_cumulative.png', dpi=300)
plt.close()

# GPU utilization

plt.figure(figsize=(10, 6))
avg_gpu = []
for model_name, data in metrics_data.items():
    if 'gpu_utilization_percent' in data:
        avg_gpu.append(np.mean(data['gpu_utilization_percent']))
    else:
        avg_gpu.append(0)

max_index = avg_gpu.index(min(avg_gpu))
colors = ['blue'] * len(avg_gpu)
colors[max_index] = 'green'

sns.barplot(x=model_names, y=avg_gpu, palette=colors)
for index, value in enumerate(avg_gpu):
    plt.text(index, value, f'{value:.4f}', ha='center', va='bottom')
plt.xlabel('Model')
plt.ylabel('Average GPU Utilization (%)')
plt.title('Average GPU Utilization')
plt.tight_layout()
plt.savefig('plots/avg_gpu_utilization.png', dpi=300)
plt.close()

plt.figure(figsize=(10, 6))
for i, (model_name, data) in enumerate(metrics_data.items()):
    if model_name == "Baseline":
        if 'gpu_utilization_percent' in data:
            plt.plot(data['gpu_utilization_percent'], label=model_name, color=colors_diff[i])
plt.xlabel('Epoch')
plt.ylabel('GPU Utilization (%)')
plt.title('GPU Utilization per Epoch. Baseline')
plt.tight_layout()
plt.savefig('plots/gpu_utilization_baseline.png', dpi=300)
plt.close()

import pandas as pd

window = 15

plt.figure(figsize=(10, 6))
for i, (model_name, data) in enumerate(metrics_data.items()):
    if model_name == "Pruned":
        if 'gpu_utilization_percent' in data:
            gpu_util = pd.Series(data['gpu_utilization_percent'])
            gpu_util_smooth = gpu_util.rolling(window, min_periods=1, center=True).mean()
            plt.plot(gpu_util_smooth, label=f"{model_name} (MA{window})", color=colors_diff[i], linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('GPU Utilization (%)')
plt.title('GPU Utilization per Epoch (Smoothed) — Pruned')
plt.tight_layout()
plt.legend()
plt.savefig('plots/gpu_utilization_pruned.png', dpi=300)
plt.close()

plt.figure(figsize=(10, 6))
for i, (model_name, data) in enumerate(metrics_data.items()):
    if model_name == "Dynamic Sparse":
        if 'gpu_utilization_percent' in data:
            gpu_util = pd.Series(data['gpu_utilization_percent'])
            gpu_util_smooth = gpu_util.rolling(window, min_periods=1, center=True).mean()
            plt.plot(gpu_util_smooth, label=f"{model_name} (MA{window})", color=colors_diff[i], linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('GPU Utilization (%)')
plt.title('GPU Utilization per Epoch (Smoothed) — Dynamic Sparse')
plt.tight_layout()
plt.legend()
plt.savefig('plots/gpu_utilization_dynamic_sparse.png', dpi=300)
plt.close()

plt.figure(figsize=(10, 6))
for i, (model_name, data) in enumerate(metrics_data.items()):
    if model_name == "Schedule Free SimCLR":
        if 'gpu_utilization_percent' in data:
            gpu_util = pd.Series(data['gpu_utilization_percent'])
            gpu_util_smooth = gpu_util.rolling(window, min_periods=1, center=True).mean()
            plt.plot(gpu_util_smooth, label=f"{model_name} (MA{window})", color=colors_diff[i], linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('GPU Utilization (%)')
plt.title('GPU Utilization per Epoch (Smoothed) — Schedule Free SimCLR')
plt.tight_layout()
plt.legend()
plt.savefig('plots/gpu_utilization_schedule_free_simclr.png', dpi=300)
plt.close()

plt.figure(figsize=(10, 6))
for i, (model_name, data) in enumerate(metrics_data.items()):
    if model_name == "DCL":
        if 'gpu_utilization_percent' in data:
            gpu_util = pd.Series(data['gpu_utilization_percent'])
            gpu_util_smooth = gpu_util.rolling(window, min_periods=1, center=True).mean()
            plt.plot(gpu_util_smooth, label=f"{model_name} (MA{window})", color=colors_diff[i], linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('GPU Utilization (%)')
plt.title('GPU Utilization per Epoch (Smoothed) — DCL')
plt.tight_layout()
plt.legend()
plt.savefig('plots/gpu_utilization_dcl.png', dpi=300)
plt.close()

print("All plots have been saved to the 'plots' directory.")
