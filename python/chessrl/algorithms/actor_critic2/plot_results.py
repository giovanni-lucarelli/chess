import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def moving_average(data, window_size=100):
    return data.rolling(window=window_size, min_periods=1).mean()

df = pd.read_csv('output/actor_critic2_dtz_3_results.csv')

window_size = 1000

fig, axes = plt.subplots(3, 1, figsize=(12, 10))

axes[0].plot(df.index, df['loss'], alpha=0.3, color='blue', label='Raw Loss')
axes[0].plot(df.index, moving_average(df['loss'], window_size), color='blue', linewidth=2, label=f'Moving Average (window={window_size})')
axes[0].set_title('Loss over Time')
axes[0].set_xlabel('Episode')
axes[0].set_ylabel('Loss')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(df.index, df['reward'], alpha=0.3, color='green', label='Raw Reward')
axes[1].plot(df.index, moving_average(df['reward'], window_size), color='green', linewidth=2, label=f'Moving Average (window={window_size})')
axes[1].set_title('Reward over Time')
axes[1].set_xlabel('Episode')
axes[1].set_ylabel('Reward')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

axes[2].plot(df.index, df['episode_return_discounted'], alpha=0.3, color='red', label='Raw Episode Return')
axes[2].plot(df.index, moving_average(df['episode_return_discounted'], window_size), color='red', linewidth=2, label=f'Moving Average (window={window_size})')
axes[2].set_title('Episode Return (Discounted) over Time')
axes[2].set_xlabel('Episode')
axes[2].set_ylabel('Episode Return (Discounted)')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('output/training_results_plots_dtz3.png', dpi=300, bbox_inches='tight')
plt.show()

print("Plots saved as 'output/training_results_plots_dtz3.png'")