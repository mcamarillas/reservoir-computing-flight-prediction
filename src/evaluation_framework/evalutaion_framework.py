import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from src.data_extractor.data_preparer import ModelType
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score

def map_predictions_to_icao(predictions, icao_info, model_type: ModelType, window_size: int=0, mean: int=0, std: int=1):
    mapped_data = {}
    
    def transform_val(val):
        return (val * std) + mean

    if model_type == ModelType.RESERVOIR_COMPUTING:
        for icao, pred in zip(icao_info, predictions):
            flat_pred = pred.flatten()
            aligned_pred = flat_pred[window_size:]
            mapped_data[icao] = transform_val(aligned_pred)
            
    elif model_type == ModelType.LSTM:
        current_idx = 0
        for icao, count in icao_info:
            segment = predictions[current_idx : current_idx + count]
            mapped_data[icao] = transform_val(segment)
            current_idx += count
            
    return mapped_data

def plot_large_residual_histograms(y_true, predictions_dict, n_bins=100, error_range=None):
    plt.figure(figsize=(14, 8))
    
    y_true_arr = np.asanyarray(y_true).flatten()
    
    print(f"{'Algorithm':<20} | {'Mean Error':<15} | {'Std Dev':<15}")
    print("-" * 55)
    
    for model_name, y_pred in predictions_dict.items():
        y_pred_arr = np.asanyarray(y_pred).flatten()
        residuals = y_pred_arr - y_true_arr
        
        mean_err = np.mean(residuals)
        std_err = np.std(residuals)
        print(f"{model_name:<20} | {mean_err:<15.4f} | {std_err:<15.4f}")
        
        sns.histplot(
            residuals, 
            bins=n_bins, 
            label=model_name,
            kde=True,
            element="step",
            stat="density",
            fill=True,
            alpha=0.2,
            binrange=error_range
        )

    plt.axvline(x=0, color='red', linestyle='-', alpha=0.6, label='Zero Error')
    plt.title(f'Residual Distribution Comparison (N={len(y_true_arr):,})', fontsize=16)
    plt.xlabel('Residual Value (Pred - True)', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def plot_icao_results(icao_id, true_map, model_map_dict, window_size):
    plt.figure(figsize=(14, 7))
        
    gt = true_map[icao_id].flatten()
    plt.plot(gt, label="True Velocity", color='black', linewidth=2, alpha=0.7)

    naive_pred = np.full_like(gt, np.nan)
    naive_pred[window_size:] = gt[:-window_size]
    
    mask = ~np.isnan(naive_pred)
    n_mape = mean_absolute_percentage_error(gt[mask], naive_pred[mask]) * 100
    n_mse = mean_squared_error(gt[mask], naive_pred[mask])
    n_r2 = r2_score(gt[mask], naive_pred[mask])
    
    plt.plot(naive_pred, label=f"Naive (MAPE: {n_mape:.2f}%)", 
             color='gray', linestyle='--', alpha=0.5)

    print(f"\n--- Statistics for ICAO: {icao_id} ---")
    print(f"Naive -> MSE: {n_mse:.4f}, MAPE: {n_mape:.2f}%, R2: {n_r2:.2f}")

    line_styles = ['-', '--', '-.', ':']
    for i, (model_name, results_dict) in enumerate(model_map_dict.items()):
        if icao_id not in results_dict:
            continue
            
        preds = results_dict[icao_id].flatten()
        min_len = min(len(gt), len(preds))
        y_true_clip = gt[:min_len]
        y_pred_clip = preds[:min_len]
        
        m_mape = mean_absolute_percentage_error(y_true_clip, y_pred_clip) * 100
        m_mse = mean_squared_error(y_true_clip, y_pred_clip)
        m_r2 = r2_score(y_true_clip, y_pred_clip)
        
        style = line_styles[i % len(line_styles)]
        plt.plot(preds, label=f"{model_name} (MAPE: {m_mape:.2f}%)", linestyle=style)
        
        print(f"{model_name} -> MSE: {m_mse:.4f}, MAPE: {m_mape:.2f}%, R2: {m_r2:.2f}")

    plt.title(f"Velocity Prediction Comparison: {icao_id}", fontsize=14)
    plt.xlabel("Time Step (Aligned)", fontsize=12)
    plt.ylabel("Velocity (Inverse Transformed)", fontsize=12)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()