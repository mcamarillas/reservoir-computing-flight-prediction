import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
from src.utils.config import ModelType, bucket
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
from src.utils.s3_io import read_json_from_s3, save_plot_to_s3
from scipy.stats import gaussian_kde

class EvaluationFramework():
    def __init__(self, transformation_file: str, target_column: str, experiment_name: str, window_size: int = 12):
        transformation_dict = read_json_from_s3(bucket, transformation_file)
        self.transformation = transformation_dict[target_column]
        self.target_column = target_column
        self.window_size = window_size
        self.experiment_name = experiment_name

    def _transform_val(self, val):
        if self.transformation["transformation"] == "std":  
            return (val * self.transformation["std"]) + self.transformation["mean"]
        return val

    def map_predictions_to_icao(self, predictions, icao_info, model_type: ModelType):
        mapped_data = {}
        if model_type == ModelType.RESERVOIR_COMPUTING:
            for icao, pred in zip(icao_info, predictions):
                flat_pred = pred.flatten()
                aligned_pred = flat_pred[self.window_size:]
                mapped_data[icao] = self._transform_val(aligned_pred)
                
        elif model_type == ModelType.LSTM:
            current_idx = 0
            for icao, count in icao_info:
                segment = predictions[current_idx : current_idx + count]
                mapped_data[icao] = self._transform_val(segment)
                current_idx += count
        return mapped_data

    def plot_icao_results(self, icao_id, true_map, model_map_dict, title_suffix=""):
        fig, ax = plt.subplots(figsize=(12, 5))
            
        y_true_flat = true_map[icao_id].flatten()
        ax.plot(y_true_flat, label=f"real {self.target_column}", color='black', linewidth=2, alpha=0.8)

        naive_pred = np.full_like(y_true_flat, np.nan)
        naive_pred[self.window_size:] = y_true_flat[:-self.window_size]
        mask = ~np.isnan(naive_pred)

        if np.any(mask):
            naive_mape = mean_absolute_percentage_error(y_true_flat[mask], naive_pred[mask]) * 100
            ax.plot(naive_pred, label=f"Naive (MAPE: {naive_mape:.1f}%)", color='gray', linestyle='--', alpha=0.6)

        line_styles = ['-.', '--', '-', ':']
        for i, (model_name, results_dict) in enumerate(model_map_dict.items()):
            if icao_id not in results_dict:
                continue
                
            preds = results_dict[icao_id].flatten()
            min_len = min(len(y_true_flat), len(preds))
            model_mape = mean_absolute_percentage_error(y_true_flat[:min_len], preds[:min_len]) * 100
            
            style = line_styles[i % len(line_styles)]
            ax.plot(preds, label=f"{model_name} ({model_mape:.1f}%)", linestyle=style, lw=1.5)

        title = f"Flight Analysis: {icao_id} {title_suffix}"
        ax.set_title(f"{title}\n{self.experiment_name}", fontsize=12, fontweight='bold')
        ax.set_ylabel(f"{self.target_column}")
        ax.set_xlabel("Time Steps")
        ax.grid(True, linestyle=':', alpha=0.5)
        ax.legend(fontsize=9, loc='upper right', frameon=True)
        plt.tight_layout()
        
        save_plot_to_s3(fig, bucket, os.path.join("experiments", self.experiment_name, "icao_results", title))
        plt.show()

    def plot_acceleration_vs_residuals(self, y_true, predictions_dict, acc_range=None, title_label="Full Range"):
        number_models = len(predictions_dict)
        fig = plt.figure(figsize=(6 * number_models + 2, 6))
        gs = fig.add_gridspec(6, (8 * number_models) + 2, hspace=0.1, wspace=0.2)

        y_true_flat = self._transform_val(np.asanyarray(y_true).flatten())
        acceleration_full = np.diff(y_true_flat)

        if acc_range is not None:
            mask = (acceleration_full >= acc_range[0]) & (acceleration_full <= acc_range[1])
            acc_viz = acceleration_full[mask]
        else:
            mask = np.ones_like(acceleration_full, dtype=bool)
            acc_viz = acceleration_full

        ax_dist_y = fig.add_subplot(gs[1:6, -2:]) 
        main_axes = []

        for i, (model_name, y_pred) in enumerate(predictions_dict.items()):
            col_start = i * 8
            ax_main = fig.add_subplot(gs[1:6, col_start : col_start+7])
            main_axes.append(ax_main)

            y_pred_flat = self._transform_val(np.asanyarray(y_pred).flatten())
            residuals = (y_pred_flat[1:] - y_true_flat[1:])[mask]
            color = plt.cm.tab10(i % 10)

            ax_main.scatter(acc_viz, residuals, alpha=0.15, s=8, color=color)
            ax_main.axhline(0, color='black', lw=1, ls='--')
            ax_main.set_title(f"{model_name}", fontsize=11)
            ax_main.set_xlabel("Acceleration")
            if i == 0: ax_main.set_ylabel("Residual (Pred - True)")

            if len(residuals) > 1 and np.var(residuals) > 1e-9:
                kde_y = gaussian_kde(residuals)
                y_space = np.linspace(residuals.min(), residuals.max(), 200)
                ax_dist_y.plot(kde_y(y_space), y_space, color=color, lw=2, label=model_name)
                ax_dist_y.fill_betweenx(y_space, 0, kde_y(y_space), color=color, alpha=0.05)

        all_y_min = min(ax.get_ylim()[0] for ax in main_axes)
        all_y_max = max(ax.get_ylim()[1] for ax in main_axes)
        for ax in main_axes: ax.set_ylim(all_y_min, all_y_max)
        ax_dist_y.set_ylim(all_y_min, all_y_max)
        ax_dist_y.axis('off')

        title = f"Residual Analysis: {title_label}"
        fig.suptitle(f"{title}\n{self.experiment_name}", fontsize=14, fontweight='bold', y=1.02)
        save_plot_to_s3(fig, bucket, os.path.join("experiments", self.experiment_name, "acc_vs_residuals", title))
        plt.show()

    def get_detailed_rankings(self, model_name, model_map_dict, true_map):
        predictions_dict = model_map_dict.get(model_name)
        if not predictions_dict: return None, None, None

        stats = []
        for icao, y_pred in predictions_dict.items():
            if icao in true_map:
                y_true, y_p = true_map[icao].flatten(), y_pred.flatten()
                min_len = min(len(y_true), len(y_p))
                mape = mean_absolute_percentage_error(y_true[:min_len], y_p[:min_len]) * 100
                mse = mean_squared_error(y_true[:min_len], y_p[:min_len])
                r2 = r2_score(y_true[:min_len], y_p[:min_len])
                stats.append({"icao24": icao, "MAPE (%)": mape, "MSE": mse, "R2": r2})

        df = pd.DataFrame(stats)
        return (df.sort_values("MAPE (%)"), df.sort_values("MSE"), df.sort_values("R2", ascending=False))

    def run_full_report(self, model_map_dict, model_pred_dict, true_map, y_true_all):
        for model_name in list(model_map_dict.keys())[:2]:
            best_mape, _, best_r2 = self.get_detailed_rankings(model_name, model_map_dict, true_map)
            
            self.plot_icao_results(best_r2.iloc[0]['icao24'], true_map, model_map_dict, 
                                   title_suffix=f"(Best R2 for {model_name})")
            
            self.plot_icao_results(best_mape.iloc[0]['icao24'], true_map, model_map_dict, 
                                   title_suffix=f"(Best MAPE for {model_name})")

        random_icaos = np.random.choice(list(true_map.keys()), 4, replace=False)
        for icao in random_icaos:
            self.plot_icao_results(icao, true_map, model_map_dict, title_suffix="(Random Sample)")

        configs = [
            (None, "Full Range"), 
            ((-10, 10), "Stable Velocity ([-10, 10])"), 
            ((-300, -10), "Deceleration ([-300, -10])"), 
            ((10, 300), "Acceleration ([10, 300])")
        ]
        
        for rng, label in configs:
            self.plot_acceleration_vs_residuals(y_true_all, model_pred_dict, 
                                                acc_range=rng, title_label=label)