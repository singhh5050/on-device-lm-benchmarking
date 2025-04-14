import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Create directory for saving plots if it doesn't exist
save_dir = 'visualizations/prefill_results'
os.makedirs(save_dir, exist_ok=True)

# -----------------------------------------------------------------------------
# Load dataset from CSV into a DataFrame
df = pd.read_csv('results/ollama_prefill_results.csv')

# Quick inspection of the data
print("Data head:")
print(df.head())
print("Unique models:", df['model'].unique())
print("Unique prefill lengths:", df['input_tokens'].unique())

# Set a clean, consistent seaborn style
sns.set_theme(style="whitegrid")

# -----------------------------------------------------------------------------
# Visualization 1: Latency vs. Prefill Length (Line Plot by Model)
plt.figure()
sns.lineplot(data=df, x='input_tokens', y='total_time_s', hue='model', marker='o')
plt.title('Prefill Length vs. Latency for Different Models')
plt.xlabel('Prefill Length')
plt.ylabel('Total Time (s)')
plt.legend(title='Model')
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'prefill_vs_latency.png'), dpi=300)
plt.show()


# -----------------------------------------------------------------------------
# Visualization 2: Prefill Tokens Per Second vs. Prefill Length (Line Plot by Model)
plt.figure(figsize=(10, 6))
sns.lineplot(data=df, x='input_tokens', y='prompt_tps', hue='model', marker='o')
plt.title('Prefill Length vs. Prefill Tokens per Second for Different Models')
plt.xlabel('Prefill Length')
plt.ylabel('Prefill Tokens per Second')
plt.legend(title='Model')
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'input_length_vs_prefill_tokens.png'), dpi=300)
plt.show()

# -----------------------------------------------------------------------------
# Visualization 3: Generation Tokens Per Second vs. Prefill Length (Line Plot by Model)
plt.figure(figsize=(10, 6))
sns.lineplot(data=df, x='input_tokens', y='generation_tps', hue='model', marker='o')
plt.title('Prefill Length vs. Generation Tokens per Second for Different Models')
plt.xlabel('Prefill Length')
plt.ylabel('Generation Tokens per Second')
plt.legend(title='Model')
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'input_length_vs_generation_tokens.png'), dpi=300)
plt.show()

# -----------------------------------------------------------------------------
# Visualization 3: Energy Consumption vs. Prefill Length

# Plot 1: On-Device Energy**

plt.figure(figsize=(10, 6))
sns.lineplot(data=df, 
             x='input_tokens', y='on_device_energy_J', 
             hue='model')
plt.title('Prefill Length vs. On-Device Energy')
plt.xlabel('Prefill Length')
plt.ylabel('Energy (Joules)')
plt.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'prefill_vs_on_device_energy.png'), dpi=300)
plt.show()

# Plot 2: Cloud Energy**

plt.figure(figsize=(10, 6))
sns.lineplot(data=df, 
             x='input_tokens', y='cloud_energy_J', 
             hue='model')
plt.title('Prefill Length vs. Cloud Energy')
plt.xlabel('Prefill Length')
plt.ylabel('Energy (Joules)')
plt.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'prefill_vs_cloud_energy.png'), dpi=300)
plt.show()


# -----------------------------------------------------------------------------
# Optional: Facet Grid for a Deep Dive per Model
# Creating a grid of latency plots by model to see individual trends more clearly.
g = sns.FacetGrid(df, col='model', height=4, aspect=1.5)
g.map(sns.lineplot, 'input_tokens', 'total_time_s', marker='o')
g.set_axis_labels("Prefill Length", "Latency (ms)")
g.add_legend()
plt.subplots_adjust(top=0.85)
g.fig.suptitle('Prefill Length vs. Latency by Model')
plt.savefig(os.path.join(save_dir, 'prefill_vs_latency_by_model.png'), dpi=300)
plt.show()
