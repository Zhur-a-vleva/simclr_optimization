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
    os.path.join("..", 'metrics/metrics_pruned.json'),
    os.path.join("..", 'metrics/metrics_schedule_free_simclr.json')
]
model_names = ['Baseline', 'DCL', 'Dynamic Sparse', 'Pruned', 'Schedule Free SimCLR']

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
colors_diff = ['blue', 'green', 'red', 'purple', 'orange']
plt.style.use('seaborn-v0_8-darkgrid')

# Linear evaluation accuracy
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

# Inference time sec
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

# Contrastive Loss
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

# Memory usage
plt.figure(figsize=(10, 6))
for i, (model_name, data) in enumerate(metrics_data.items()):
    if 'memory_usage_MB' in data:
        plt.plot(data['memory_usage_MB'], label=model_name, color=colors_diff[i])
plt.xlabel('Epoch')
plt.ylabel('Memory Usage (MB)')
plt.title('Memory Usage per Epoch')
plt.legend()
plt.tight_layout()
plt.savefig('plots/memory_usage.png', dpi=600)
plt.close()

# Model size
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

# Training time per epoch
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
        else:
            cumulative_pruned = sum(data['training_time_per_epoch_sec'])
        plt.plot(data['training_time_per_epoch_sec'], label=model_name, color=colors_diff[i])
plt.xlabel('Epoch')
plt.ylabel('Training Time (seconds)')
plt.title('Training Time per Epoch (without Dynamic Sparse)')
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


#
# # 5. GPU utilization
# plt.figure(figsize=(10, 6))
# for i, (model_name, data) in enumerate(metrics_data.items()):
#     if 'gpu_utilization_percent' in data:
#         plt.plot(data['gpu_utilization_percent'], label=model_name, color=colors[i])
# plt.xlabel('Epoch')
# plt.ylabel('GPU Utilization (%)')
# plt.title('GPU Utilization per Epoch')
# plt.legend()
# plt.tight_layout()
# plt.savefig('plots/gpu_utilization.png', dpi=300)
# plt.close()
#
# # 6. Average GPU utilization (bar chart)
# plt.figure(figsize=(10, 6))
# avg_gpu = []
# for model_name, data in metrics_data.items():
#     if 'gpu_utilization_percent' in data:
#         avg_gpu.append(np.mean(data['gpu_utilization_percent']))
#     else:
#         avg_gpu.append(0)
#
# sns.barplot(x=model_names, y=avg_gpu, palette=colors)
# plt.xlabel('Model')
# plt.ylabel('Average GPU Utilization (%)')
# plt.title('Average GPU Utilization')
# plt.tight_layout()
# plt.savefig('plots/avg_gpu_utilization.png', dpi=300)
# plt.close()
#
# # 7. GPU utilization comparison: first 50 vs last 50 epochs
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
#
# for i, (model_name, data) in enumerate(metrics_data.items()):
#     if 'gpu_utilization_percent' in data:
#         gpu_util = data['gpu_utilization_percent']
#         if len(gpu_util) >= 50:
#             # First 50 epochs
#             ax1.plot(gpu_util[:50], label=model_name, color=colors[i])
#             # Last 50 epochs
#             ax2.plot(range(len(gpu_util)-50, len(gpu_util)), gpu_util[-50:], label=model_name, color=colors[i])
#         else:
#             # If we have less than 50 epochs, use all available data
#             ax1.plot(gpu_util, label=model_name, color=colors[i])
#             ax2.plot([], [], label=model_name, color=colors[i])  # Empty plot for consistency
#
# ax1.set_xlabel('Epoch')
# ax1.set_ylabel('GPU Utilization (%)')
# ax1.set_title('First 50 Epochs')
# ax1.legend()
#
# ax2.set_xlabel('Epoch')
# ax2.set_ylabel('GPU Utilization (%)')
# ax2.set_title('Last 50 Epochs')
# ax2.legend()
#
# plt.tight_layout()
# plt.savefig('plots/gpu_utilization_comparison.png', dpi=300)
# plt.close()
#

print("All plots have been saved to the 'plots' directory.")
