import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_large_residual_histograms(y_true, predictions_dict, n_bins=100, error_range=None):
    plt.figure(figsize=(14, 8))
    
    y_true_arr = np.asanyarray(y_true).flatten()
    
    # Print header for the statistics
    print(f"{'Algorithm':<20} | {'Mean Error':<15} | {'Std Dev':<15}")
    print("-" * 55)
    
    for model_name, y_pred in predictions_dict.items():
        y_pred_arr = np.asanyarray(y_pred).flatten()
        residuals = y_pred_arr - y_true_arr
        
        # Calculate stats
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

    plt.axvline(x=0, color='red', linestyle='--', alpha=0.6, label='Zero Error')
    plt.title(f'Residual Distribution Comparison (N={len(y_true_arr):,})', fontsize=16)
    plt.xlabel('Residual Value (Pred - True)', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()