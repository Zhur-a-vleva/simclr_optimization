import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Создаем директорию для сохранения графиков, если она не существует
this_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = this_dir

# Предположим, что у нас есть доступ к метрикам обеих моделей после их обучения
# baseline_metrics и pruned_metrics - это словари с результатами обучения

def create_comparison_plots(baseline_metrics, pruned_metrics):
    """
    Создает и сохраняет графики сравнения метрик для двух моделей

    Args:
        baseline_metrics (dict): Метрики базовой модели
        pruned_metrics (dict): Метрики модели с прунингом
    """

    # 1. График обучающей функции потерь
    plt.figure(figsize=(12, 6))
    plt.plot(baseline_metrics["contrastive_loss"], label="Baseline Model")
    plt.plot(pruned_metrics["contrastive_loss"], label="Pruned Model")
    plt.xlabel("Epoch")
    plt.ylabel("Contrastive Loss")
    plt.title("Training Loss Comparison")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.savefig(f"{output_dir}/training_loss_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 2. График времени обучения на эпоху
    plt.figure(figsize=(12, 6))
    plt.plot(baseline_metrics["training_time_per_epoch_sec"], label="Baseline Model")
    plt.plot(pruned_metrics["training_time_per_epoch_sec"], label="Pruned Model")
    plt.xlabel("Epoch")
    plt.ylabel("Training Time (seconds)")
    plt.title("Training Time per Epoch Comparison")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.savefig(f"{output_dir}/training_time_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 3. График использования GPU
    plt.figure(figsize=(12, 6))
    plt.plot(baseline_metrics["gpu_utilization_percent"], label="Baseline Model")
    plt.plot(pruned_metrics["gpu_utilization_percent"], label="Pruned Model")
    plt.xlabel("Epoch")
    plt.ylabel("GPU Utilization (%)")
    plt.title("GPU Utilization Comparison")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.savefig(f"{output_dir}/gpu_utilization_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 4. График использования памяти
    plt.figure(figsize=(12, 6))
    plt.plot(baseline_metrics["memory_usage_MB"], label="Baseline Model")
    plt.plot(pruned_metrics["memory_usage_MB"], label="Pruned Model")
    plt.xlabel("Epoch")
    plt.ylabel("Memory Usage (MB)")
    plt.title("Memory Usage Comparison")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.savefig(f"{output_dir}/memory_usage_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 5. График размера модели
    plt.figure(figsize=(12, 6))
    plt.plot(baseline_metrics["model_size_MB"], label="Baseline Model")
    plt.plot(pruned_metrics["model_size_MB"], label="Pruned Model")
    plt.xlabel("Epoch")
    plt.ylabel("Model Size (MB)")
    plt.title("Model Size Comparison")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.savefig(f"{output_dir}/model_size_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 6. Сводная статистика на последней эпохе (барчарт)
    metrics_names = ["Final Loss", "Training Time (s)", "GPU Util (%)", "Memory (MB)", "Model Size (MB)"]
    baseline_values = [
        baseline_metrics["contrastive_loss"][-1],
        baseline_metrics["training_time_per_epoch_sec"][-1],
        baseline_metrics["gpu_utilization_percent"][-1],
        baseline_metrics["memory_usage_MB"][-1],
        baseline_metrics["model_size_MB"][-1]
    ]
    pruned_values = [
        pruned_metrics["contrastive_loss"][-1],
        pruned_metrics["training_time_per_epoch_sec"][-1],
        pruned_metrics["gpu_utilization_percent"][-1],
        pruned_metrics["memory_usage_MB"][-1],
        pruned_metrics["model_size_MB"][-1]
    ]

    # Создаем DataFrame для удобства построения
    df = pd.DataFrame({
        'Metric': metrics_names * 2,
        'Value': baseline_values + pruned_values,
        'Model': ['Baseline'] * 5 + ['Pruned'] * 5
    })

    plt.figure(figsize=(14, 8))
    sns.barplot(x='Metric', y='Value', hue='Model', data=df)
    plt.title("Final Metrics Comparison")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/final_metrics_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 7. График относительного улучшения для каждой метрики
    improvement_percentages = []
    for i in range(len(metrics_names)):
        if metrics_names[i] == "Final Loss":
            # Для функции потерь, меньше - лучше
            improv = (baseline_values[i] - pruned_values[i]) / (baseline_values[i] * 100 + 1)
        else:
            # Для остальных метрик (время, память, размер), меньше - лучше
            improv = (baseline_values[i] - pruned_values[i]) / (baseline_values[i] * 100 + 1)
        improvement_percentages.append(improv)

    plt.figure(figsize=(12, 6))
    bars = plt.bar(metrics_names, improvement_percentages, color='skyblue')

    # Раскрашиваем отрицательные значения красным
    for i, bar in enumerate(bars):
        if improvement_percentages[i] < 0:
            bar.set_color('salmon')

    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.ylabel("Improvement (%)")
    plt.title("Relative Improvement of Pruned Model")
    plt.xticks(rotation=30)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Добавляем метки значений над столбцами
    for i, v in enumerate(improvement_percentages):
        plt.text(i, v + (5 if v >= 0 else -5),
                 f"{v:.1f}%",
                 ha='center',
                 va='bottom' if v >= 0 else 'top',
                 fontweight='bold')

    plt.tight_layout()
    plt.savefig(f"{output_dir}/relative_improvement.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 8. График сравнения эффективности (соотношение качество/ресурсы)
    # Определяем эффективность как отношение обратной величины потерь к использованию ресурсов
    baseline_efficiency = 1 / (baseline_values[0] * baseline_values[4])  # 1/(loss * model_size)
    pruned_efficiency = 1 / (pruned_values[0] * pruned_values[4])  # 1/(loss * model_size)

    plt.figure(figsize=(10, 6))
    plt.bar(['Baseline Model', 'Pruned Model'], [baseline_efficiency, pruned_efficiency], color=['blue', 'green'])
    plt.ylabel('Efficiency (1 / (Loss * Model Size))')
    plt.title('Model Efficiency Comparison')

    # Добавляем метки значений
    for i, v in enumerate([baseline_efficiency, pruned_efficiency]):
        plt.text(i, v / 2, f"{v:.2e}", ha='center', color='white', fontweight='bold')

    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(f"{output_dir}/efficiency_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()


# Функция для загрузки метрик, если они сохранены в файлы
def load_metrics_from_files(baseline_file, pruned_file):
    import json

    with open(baseline_file, 'r') as f:
        baseline_metrics = json.load(f)

    with open(pruned_file, 'r') as f:
        pruned_metrics = json.load(f)

    return baseline_metrics, pruned_metrics



# Если метрики доступны в виде экземпляров класса Metrics
def create_plots_from_metrics_objects(baseline_metrics_obj, pruned_metrics_obj):
    baseline_metrics = baseline_metrics_obj.metrics
    pruned_metrics = pruned_metrics_obj.metrics
    create_comparison_plots(baseline_metrics, pruned_metrics)


if __name__ == "__main__":
    baseline_metrics, pruned_metrics = load_metrics_from_files(os.path.join("..", "metrics/metrics_dcl.json"),
                                                               os.path.join("..", "metrics/metrics_pruned.json"))
    create_comparison_plots(baseline_metrics, pruned_metrics)