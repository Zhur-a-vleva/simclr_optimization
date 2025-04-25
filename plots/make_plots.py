import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import json

# Создаем директорию для сохранения графиков, если она не существует

# Create plots directory if it doesn't exist
os.makedirs('plots', exist_ok=True)

# Define file paths and model names
file_paths = [
    os.path.join("..", 'metrics/metrics_baseline.json'),
    os.path.join("..", "metrics/metrics_dcl.json"),
    os.path.join("..", 'metrics/metrics_dynamic_sparse.json'),
    os.path.join("..", 'metrics/metrics_pruned.json')
]
model_names = ['Baseline', 'DCL', 'Dynamic Sparse', 'Pruned']

# Load metrics data
metrics_data = {}
for file_path, model_name in zip(file_paths, model_names):
    try:
        with open(file_path, 'r') as f:
            metrics_data[model_name] = json.load(f)
    except FileNotFoundError:
        print(f"Warning: {file_path} not found!")
        metrics_data[model_name] = {}

# Set up a consistent color scheme
colors = ['blue', 'green', 'red', 'purple']
plt.style.use('seaborn-v0_8-darkgrid')

# 1. Loss function plot
plt.figure(figsize=(10, 6))
for i, (model_name, data) in enumerate(metrics_data.items()):
    if 'contrastive_loss' in data:
        plt.plot(data['contrastive_loss'], label=model_name, color=colors[i])
plt.xlabel('Epoch')
plt.ylabel('Contrastive Loss')
plt.title('Training Loss Comparison')
plt.legend()
plt.tight_layout()
plt.savefig('plots/contrastive_loss.png', dpi=300)
plt.close()

# 2. Training time per epoch
plt.figure(figsize=(10, 6))
for i, (model_name, data) in enumerate(metrics_data.items()):
    if 'training_time_per_epoch_sec' in data:
        plt.plot(data['training_time_per_epoch_sec'], label=model_name, color=colors[i])
plt.xlabel('Epoch')
plt.ylabel('Training Time (seconds)')
plt.title('Training Time per Epoch')
plt.legend()
plt.tight_layout()
plt.savefig('plots/training_time.png', dpi=300)
plt.close()

# 3. Average training time per epoch (bar chart)
plt.figure(figsize=(10, 6))
avg_times = []
for model_name, data in metrics_data.items():
    if 'training_time_per_epoch_sec' in data:
        avg_times.append(np.mean(data['training_time_per_epoch_sec']))
    else:
        avg_times.append(0)

sns.barplot(x=model_names, y=avg_times, palette=colors)
plt.xlabel('Model')
plt.ylabel('Average Training Time (seconds)')
plt.title('Average Training Time per Epoch')
plt.tight_layout()
plt.savefig('plots/avg_training_time.png', dpi=300)
plt.close()

# 4. Training time comparison: first 50 vs last 50 epochs
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

for i, (model_name, data) in enumerate(metrics_data.items()):
    if 'training_time_per_epoch_sec' in data:
        times = data['training_time_per_epoch_sec']
        if len(times) >= 50:
            # First 50 epochs
            ax1.plot(times[:50], label=model_name, color=colors[i])
            # Last 50 epochs
            ax2.plot(range(len(times)-50, len(times)), times[-50:], label=model_name, color=colors[i])
        else:
            # If we have less than 50 epochs, use all available data
            ax1.plot(times, label=model_name, color=colors[i])
            ax2.plot([], [], label=model_name, color=colors[i])  # Empty plot for consistency

ax1.set_xlabel('Epoch')
ax1.set_ylabel('Training Time (seconds)')
ax1.set_title('First 50 Epochs')
ax1.legend()

ax2.set_xlabel('Epoch')
ax2.set_ylabel('Training Time (seconds)')
ax2.set_title('Last 50 Epochs')
ax2.legend()

plt.tight_layout()
plt.savefig('plots/training_time_comparison.png', dpi=300)
plt.close()

# 5. GPU utilization
plt.figure(figsize=(10, 6))
for i, (model_name, data) in enumerate(metrics_data.items()):
    if 'gpu_utilization_percent' in data:
        plt.plot(data['gpu_utilization_percent'], label=model_name, color=colors[i])
plt.xlabel('Epoch')
plt.ylabel('GPU Utilization (%)')
plt.title('GPU Utilization per Epoch')
plt.legend()
plt.tight_layout()
plt.savefig('plots/gpu_utilization.png', dpi=300)
plt.close()

# 6. Average GPU utilization (bar chart)
plt.figure(figsize=(10, 6))
avg_gpu = []
for model_name, data in metrics_data.items():
    if 'gpu_utilization_percent' in data:
        avg_gpu.append(np.mean(data['gpu_utilization_percent']))
    else:
        avg_gpu.append(0)

sns.barplot(x=model_names, y=avg_gpu, palette=colors)
plt.xlabel('Model')
plt.ylabel('Average GPU Utilization (%)')
plt.title('Average GPU Utilization')
plt.tight_layout()
plt.savefig('plots/avg_gpu_utilization.png', dpi=300)
plt.close()

# 7. GPU utilization comparison: first 50 vs last 50 epochs
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

for i, (model_name, data) in enumerate(metrics_data.items()):
    if 'gpu_utilization_percent' in data:
        gpu_util = data['gpu_utilization_percent']
        if len(gpu_util) >= 50:
            # First 50 epochs
            ax1.plot(gpu_util[:50], label=model_name, color=colors[i])
            # Last 50 epochs
            ax2.plot(range(len(gpu_util)-50, len(gpu_util)), gpu_util[-50:], label=model_name, color=colors[i])
        else:
            # If we have less than 50 epochs, use all available data
            ax1.plot(gpu_util, label=model_name, color=colors[i])
            ax2.plot([], [], label=model_name, color=colors[i])  # Empty plot for consistency

ax1.set_xlabel('Epoch')
ax1.set_ylabel('GPU Utilization (%)')
ax1.set_title('First 50 Epochs')
ax1.legend()

ax2.set_xlabel('Epoch')
ax2.set_ylabel('GPU Utilization (%)')
ax2.set_title('Last 50 Epochs')
ax2.legend()

plt.tight_layout()
plt.savefig('plots/gpu_utilization_comparison.png', dpi=300)
plt.close()

# 8. Memory usage
plt.figure(figsize=(10, 6))
for i, (model_name, data) in enumerate(metrics_data.items()):
    if 'memory_usage_MB' in data:
        plt.plot(data['memory_usage_MB'], label=model_name, color=colors[i])
plt.xlabel('Epoch')
plt.ylabel('Memory Usage (MB)')
plt.title('Memory Usage per Epoch')
plt.legend()
plt.tight_layout()
plt.savefig('plots/memory_usage.png', dpi=300)
plt.close()

# 9. Model size
plt.figure(figsize=(10, 6))
for i, (model_name, data) in enumerate(metrics_data.items()):
    if 'model_size_MB' in data:
        plt.plot(data['model_size_MB'], label=model_name, color=colors[i])
plt.xlabel('Epoch')
plt.ylabel('Model Size (MB)')
plt.title('Model Size per Epoch')
plt.legend()
plt.tight_layout()
plt.savefig('plots/model_size.png', dpi=300)
plt.close()

# 10. Final accuracy metrics (bar charts)
# Linear evaluation accuracy
plt.figure(figsize=(10, 6))
final_accuracy = []
for model_name, data in metrics_data.items():
    if 'linear_evaluation_accuracy' in data and data['linear_evaluation_accuracy']:
        final_accuracy.append(data['linear_evaluation_accuracy'][-1])
    else:
        final_accuracy.append(0)

sns.barplot(x=model_names, y=final_accuracy, palette=colors)
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
    else:
        final_nmi.append(0)

sns.barplot(x=model_names, y=final_nmi, palette=colors)
plt.xlabel('Model')
plt.ylabel('NMI')
plt.title('Final Normalized Mutual Information (NMI)')
plt.tight_layout()
plt.savefig('plots/final_nmi.png', dpi=300)
plt.close()

print("All plots have been saved to the 'plots' directory.")
