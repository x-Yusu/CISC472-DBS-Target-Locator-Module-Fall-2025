import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Load control metrics from numpy file
control_metrics = np.load(r'..\baseline_dist.npy', allow_pickle=True)  # adjust filename as needed
print(control_metrics)

ctrl_mean = np.mean(control_metrics, axis=0)
ctrl_std = np.std(control_metrics, axis=0)

# Extract data for whole right amygdala and left caudate nucleus
data_4 = np.mean(control_metrics[:, 3:4], axis=1)
data_9 = np.mean(control_metrics[:, 25:26], axis=1)

# Create figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Histograms with fitted normal curve
axes[0, 0].hist(data_4, bins=30, alpha=0.7, edgecolor='black', density=True, label='Data')
mu_4, std_4 = data_4.mean(), data_4.std()
x_4 = np.linspace(data_4.min(), data_4.max(), 100)
axes[0, 0].plot(x_4, stats.norm.pdf(x_4, mu_4, std_4), 'r-', lw=2, label='Normal fit')
axes[0, 0].set_title(f'Distribution of right amygdala\nMean: {mu_4:.3f}, Std: {std_4:.3f}')
axes[0, 0].set_xlabel('Value')
axes[0, 0].set_ylabel('Density')
axes[0, 0].legend()

axes[0, 1].hist(data_9, bins=30, alpha=0.7, edgecolor='black', density=True, label='Data')
mu_9, std_9 = data_9.mean(), data_9.std()
x_9 = np.linspace(data_9.min(), data_9.max(), 100)
axes[0, 1].plot(x_9, stats.norm.pdf(x_9, mu_9, std_9), 'r-', lw=2, label='Normal fit')
axes[0, 1].set_title(f'Distribution of left caudate nucleus\nMean: {mu_9:.3f}, Std: {std_9:.3f}')
axes[0, 1].set_xlabel('Value')
axes[0, 1].set_ylabel('Density')
axes[0, 1].legend()

# Q-Q plots for normality check
stats.probplot(data_4, dist="norm", plot=axes[1, 0])
axes[1, 0].set_title('Q-Q Plot for amygdala')
axes[1, 0].grid(True, alpha=0.3)

stats.probplot(data_9, dist="norm", plot=axes[1, 1])
axes[1, 1].set_title('Q-Q Plot for caudate nucleus')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Normality tests
print("="*50)
print("NORMALITY TESTS")
print("="*50)

print("\nRight Amygdala:")
stat_4, p_4 = stats.shapiro(data_4)
print(f"  Shapiro-Wilk: statistic={stat_4:.4f}, p-value={p_4:.4f}")
k2_4, p_k4 = stats.normaltest(data_4)
print(f"  D'Agostino-K²: statistic={k2_4:.4f}, p-value={p_k4:.4f}")
print(f"  Interpretation: {'NORMAL' if p_4 > 0.05 else 'NOT NORMAL'} (α=0.05)")

print("\nCaudate Nucleus:")
stat_9, p_9 = stats.shapiro(data_9)
print(f"  Shapiro-Wilk: statistic={stat_9:.4f}, p-value={p_9:.4f}")
k2_9, p_k9 = stats.normaltest(data_9)
print(f"  D'Agostino-K²: statistic={k2_9:.4f}, p-value={p_k9:.4f}")
print(f"  Interpretation: {'NORMAL' if p_9 > 0.05 else 'NOT NORMAL'} (α=0.05)")