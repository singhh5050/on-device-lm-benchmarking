import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter
import os

# Create directory for saving plots if it doesn't exist
save_dir = 'visualizations/quant_results'
os.makedirs(save_dir, exist_ok=True)

# Load dataset from CSV into a DataFrame
df = pd.read_csv('results/ollama_quant_results.csv')

# Extract quantization information from the model column
# Example: 'mistral:7b-instruct-q4_0' -> 'q4_0'
df['quantization'] = df['model'].apply(lambda x: x.split('-')[-1])

# Print unique quantization values to inspect the data
print("Unique quantization values:", df['quantization'].unique())

# -----------------------------------------------------------------------------
# Create a custom ordering for the quantization column.
# The note specifies that "q4_0" is the least quantized (i.e. the baseline/latest).
# We extract the numeric component from strings like "q4_0" so that we can sort them.
def extract_bit(q):
    try:
        # For instance: "q4_0" -> 4
        return int(q.split('_')[0][1:])
    except Exception as e:
        return None

# Add a helper column with the numeric quantization level
df['bit'] = df['quantization'].apply(extract_bit)

# We want to order the quantization categories from the least quantized (largest 'bit' value)
# to the most quantized (smallest 'bit' value).
ordered_quant = df.sort_values('bit', ascending=False)['quantization'].unique().tolist()
print("Ordered quantization values:", ordered_quant)

# Set a clean seaborn style for the plots
sns.set_theme(style="whitegrid")

# -----------------------------------------------------------------------------
# Visualization 1: Quantization vs. Latency (with log scale)
plt.figure(figsize=(10, 6))
sns.boxplot(x='quantization', y='total_time_s', data=df, order=ordered_quant)
plt.title('Quantization vs. Latency')
plt.xlabel('Quantization')
plt.ylabel('Latency (s)')
plt.yscale('log')  # Set y-axis to log scale
plt.tight_layout()
# Save the plot
plt.savefig(os.path.join(save_dir, 'quantization_vs_latency.png'), dpi=300)
plt.show()

# -----------------------------------------------------------------------------
# Visualization 2: Quantization vs. On-Device Energy
plt.figure(figsize=(10, 6))
sns.boxplot(x='quantization', y='on_device_energy_J', data=df, order=ordered_quant)
plt.title('Quantization vs. On-Device Energy')
plt.xlabel('Quantization')
plt.ylabel('On-Device Energy (Joules)')
plt.tight_layout()
# Save the plot
plt.savefig(os.path.join(save_dir, 'quantization_vs_on_device.png'), dpi=300)
plt.show()

# -----------------------------------------------------------------------------
# Visualization 3: Quantization vs. Cloud Energy with explicit 1e6 notation for each tick
plt.figure(figsize=(10, 6))
sns.boxplot(x='quantization', y='cloud_energy_J', data=df, order=ordered_quant)
plt.title('Quantization vs. Cloud Energy')
plt.xlabel('Quantization')
plt.ylabel('Cloud Energy (Joules)')

# Format y-axis ticks to explicitly show 1e6 notation on each tick
def scientific_formatter(x, pos):
    return '{:.1f}e6'.format(x/1000000)
plt.gca().yaxis.set_major_formatter(FuncFormatter(scientific_formatter))

plt.tight_layout()
# Save the plot
plt.savefig(os.path.join(save_dir, 'quantization_cloud_energy.png'), dpi=300)
plt.show()