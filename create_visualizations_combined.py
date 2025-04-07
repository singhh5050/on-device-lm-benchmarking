import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Create visualization directories
os.makedirs("visualizations/mlx", exist_ok=True)
os.makedirs("visualizations/ollama", exist_ok=True)

# Set global styling for all plots
sns.set(style="whitegrid")
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
})

# Common colors for consistent styling
COLORS = {'prefill-heavy': '#3498db', 'decode-heavy': '#e74c3c'}

####################################
# MLX VISUALIZATIONS
####################################
def create_mlx_visualizations():
    print("Creating MLX visualizations...")
    
    # Read the MLX results CSV
    df = pd.read_csv("results/mlx_results.csv")
    print(f"MLX results data: {len(df)} records")
    print("Columns:", df.columns.tolist())
    
    # Clean model names for better display
    df['model_short'] = df['model'].apply(lambda x: x.split('/')[-1].replace('-4bit', ''))
    
    # Get color palette for models
    model_colors = sns.color_palette("viridis", len(df['model_short'].unique()))
    
    # Plot 1: Prompt TPS vs Generation TPS Comparison (matching Ollama style)
    plt.figure(figsize=(10, 8))
    for wtype in df['workload_type'].unique():
        sub_df = df[df['workload_type'] == wtype]
        
        # Create scatter plot
        plt.scatter(sub_df['prompt_tps'], sub_df['generation_tps'], 
                   color=COLORS[wtype], label=wtype, alpha=0.7, edgecolor='k', s=100)
        
        # Add model names as annotations
        for i, row in sub_df.iterrows():
            plt.annotate(row['model_short'], 
                        (row['prompt_tps'], row['generation_tps']),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=9)
    
    plt.xlabel("Prompt TPS (prefill tokens/sec)")
    plt.ylabel("Generation TPS (decode tokens/sec)")
    plt.title("MLX: Prompt TPS vs. Generation TPS by Workload Type")
    plt.legend()
    plt.tight_layout()
    plt.savefig("visualizations/mlx/1_prompt_vs_gen_tps.png", dpi=300)
    plt.close()
    
    # Plot 2: Workload Ratio vs Estimated Processing Time
    plt.figure(figsize=(10, 8))
    # Calculate estimated processing time
    df['estimated_time'] = df['input_tokens'] / df['prompt_tps'] + df['generated_tokens'] / df['generation_tps']
    
    for wtype in df['workload_type'].unique():
        sub_df = df[df['workload_type'] == wtype]
        plt.scatter(sub_df['workload_ratio'], sub_df['estimated_time'],
                   color=COLORS[wtype], label=wtype, alpha=0.7, edgecolor='k', s=100)
        
        # Add model names as annotations
        for i, row in sub_df.iterrows():
            plt.annotate(row['model_short'], 
                        (row['workload_ratio'], row['estimated_time']),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=9)
    
    plt.xlabel("Workload Ratio (Input Tokens / Total Tokens)")
    plt.ylabel("Estimated Processing Time (seconds)")
    plt.title("MLX: Workload Ratio vs. Estimated Processing Time")
    plt.legend()
    plt.tight_layout()
    plt.savefig("visualizations/mlx/2_workload_ratio_vs_time.png", dpi=300)
    plt.close()
    
    # Plot 3: Model Performance Comparison (by TPS)
    # Group by model and workload_type
    grouped = df.groupby(['model_short', 'workload_type']).agg({
        'prompt_tps': 'mean',
        'generation_tps': 'mean'
    }).reset_index()
    
    # Get model names without version tags for cleaner display
    models = grouped['model_short'].unique()
    model_names = models
    
    # Create the grouped bar chart
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Set up new organization by workload type first
    workload_types = ['prefill-heavy', 'decode-heavy']
    metrics = ['prompt_tps', 'generation_tps']
    workload_colors = {
        'prefill-heavy': ['#3498db', '#3498db'],  # Same blue for both prompt and generation
        'decode-heavy': ['#e74c3c', '#e74c3c']    # Same red for both prompt and generation
    }
    
    # Create the bars
    x = np.arange(len(model_names))
    bar_width = 0.2
    opacity = 0.8
    
    # Position offsets for each bar within workload group
    offsets = {
        'prefill-heavy': {'prompt_tps': -bar_width/2, 'generation_tps': bar_width/2},
        'decode-heavy': {'prompt_tps': -bar_width/2 + 2*bar_width, 'generation_tps': bar_width/2 + 2*bar_width}
    }
    
    # Plot bars
    for wtype in workload_types:
        for i, metric in enumerate(metrics):
            values = grouped[grouped['workload_type'] == wtype][metric].values
            plt.bar(x + offsets[wtype][metric], values, bar_width, alpha=opacity, 
                    color=workload_colors[wtype][i], 
                    label=f"{wtype} ({metric.replace('_tps', '')})")
    
    plt.xlabel('Model')
    plt.ylabel('Tokens Per Second (TPS)')
    plt.title('MLX: Average TPS by Workload Type and Processing Phase')
    plt.xticks(x, model_names, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig("visualizations/mlx/3_model_tps_comparison.png", dpi=300)
    plt.close()
    
    # Plot 4: Input vs Generated Tokens
    plt.figure(figsize=(10, 8))
    for idx, model in enumerate(df['model_short'].unique()):
        sub_df = df[df['model_short'] == model]
        plt.scatter(sub_df['input_tokens'], sub_df['generated_tokens'], 
                   color=model_colors[idx], label=model, alpha=0.7, edgecolor='k', s=100)
    
    plt.xlabel("Input Tokens")
    plt.ylabel("Generated Tokens")
    plt.title("MLX: Input vs. Generated Tokens by Model")
    plt.legend()
    plt.tight_layout()
    plt.savefig("visualizations/mlx/4_input_vs_generated_tokens.png", dpi=300)
    plt.close()
    
    # Plot 5: Peak memory by model and dataset
    plt.figure(figsize=(12, 8))
    sns.boxplot(x='model_short', y='peak_memory', hue='dataset', 
               data=df, palette='Set2')
    plt.xticks(rotation=45)
    plt.xlabel('Model')
    plt.ylabel('Peak Memory Usage (GB)')
    plt.title('MLX: Memory Usage by Model and Dataset')
    plt.legend(title='Dataset')
    plt.tight_layout()
    plt.savefig("visualizations/mlx/5_memory_by_model_dataset.png", dpi=300)
    plt.close()
    
    # Plot 6: Workload Ratio Distribution (keeping this plot as it's informative)
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    
    # Histogram
    sns.histplot(
        data=df, 
        x='workload_ratio',
        hue='workload_type',
        bins=20,
        kde=True,
        ax=axes[0],
        palette=COLORS
    )
    axes[0].set_title('MLX: Distribution of Workload Ratio')
    axes[0].set_xlabel('Workload Ratio (input/total)')
    axes[0].set_ylabel('Count')
    
    # Boxplot by model
    sns.boxplot(
        data=df, 
        x='model_short', 
        y='workload_ratio',
        hue='workload_type',
        ax=axes[1],
        palette=COLORS
    )
    axes[1].set_title('MLX: Workload Ratio by Model and Type')
    axes[1].set_xlabel('Model')
    axes[1].set_ylabel('Workload Ratio (input/total)')
    axes[1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig("visualizations/mlx/6_workload_ratio_distribution.png", dpi=300)
    plt.close()
    
    print("✅ MLX visualizations created successfully.")

####################################
# OLLAMA VISUALIZATIONS
####################################
def create_ollama_visualizations():
    print("Creating Ollama visualizations...")
    
    # Load the Ollama results data
    df_ollama = pd.read_csv("results/ollama_results.csv")
    print(f"Ollama results data: {len(df_ollama)} records")
    print("Columns:", df_ollama.columns.tolist())
    
    # Set up colors for workload types - use the same colors as MLX for consistency
    model_colors = sns.color_palette("viridis", len(df_ollama['model'].unique()))
    
    # Clean model names for better display
    df_ollama['model_short'] = df_ollama['model'].apply(lambda x: x.split(':')[0])
    
    # Plot 1: Prompt TPS vs Generation TPS Comparison
    plt.figure(figsize=(10, 8))
    for wtype in df_ollama['workload_type'].unique():
        sub_df = df_ollama[df_ollama['workload_type'] == wtype]
        
        # Create scatter plot
        plt.scatter(sub_df['prompt_tps'], sub_df['generation_tps'], 
                   color=COLORS[wtype], label=wtype, alpha=0.7, edgecolor='k', s=100)
        
        # Add model names as annotations
        for i, row in sub_df.iterrows():
            plt.annotate(row['model_short'], 
                        (row['prompt_tps'], row['generation_tps']),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=9)
    
    plt.xlabel("Prompt TPS (prefill tokens/sec)")
    plt.ylabel("Generation TPS (decode tokens/sec)")
    plt.title("Ollama: Prompt TPS vs. Generation TPS by Workload Type")
    plt.legend()
    plt.tight_layout()
    plt.savefig("visualizations/ollama/1_prompt_vs_gen_tps.png", dpi=300)
    plt.close()
    
    # Plot 2: Workload Ratio vs Total Processing Time
    plt.figure(figsize=(10, 8))
    for wtype in df_ollama['workload_type'].unique():
        sub_df = df_ollama[df_ollama['workload_type'] == wtype]
        plt.scatter(sub_df['workload_ratio'], sub_df['total_time_s'],
                   color=COLORS[wtype], label=wtype, alpha=0.7, edgecolor='k', s=100)
        
        # Add model names as annotations
        for i, row in sub_df.iterrows():
            plt.annotate(row['model_short'], 
                        (row['workload_ratio'], row['total_time_s']),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=9)
    
    plt.xlabel("Workload Ratio (Input Tokens / Total Tokens)")
    plt.ylabel("Total Processing Time (seconds)")
    plt.title("Ollama: Workload Ratio vs. Total Processing Time")
    plt.legend()
    plt.tight_layout()
    plt.savefig("visualizations/ollama/2_workload_ratio_vs_time.png", dpi=300)
    plt.close()
    
    # Plot 3: Model Performance Comparison (by TPS)
    # Group by model and workload_type
    grouped = df_ollama.groupby(['model', 'workload_type']).agg({
        'prompt_tps': 'mean',
        'generation_tps': 'mean'
    }).reset_index()
    
    # Get model names without version tags for cleaner display
    models = grouped['model'].unique()
    model_names = [m.split(':')[0] for m in models]
    
    # Create the grouped bar chart
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Set up new organization by workload type first
    workload_types = ['prefill-heavy', 'decode-heavy']
    metrics = ['prompt_tps', 'generation_tps']
    workload_colors = {
        'prefill-heavy': ['#3498db', '#3498db'],  # Same blue for both prompt and generation
        'decode-heavy': ['#e74c3c', '#e74c3c']    # Same red for both prompt and generation
    }
    
    # Create the bars
    x = np.arange(len(model_names))
    bar_width = 0.2
    opacity = 0.8
    
    # Position offsets for each bar within workload group
    offsets = {
        'prefill-heavy': {'prompt_tps': -bar_width/2, 'generation_tps': bar_width/2},
        'decode-heavy': {'prompt_tps': -bar_width/2 + 2*bar_width, 'generation_tps': bar_width/2 + 2*bar_width}
    }
    
    # Plot bars
    for wtype in workload_types:
        for i, metric in enumerate(metrics):
            values = grouped[grouped['workload_type'] == wtype][metric].values
            plt.bar(x + offsets[wtype][metric], values, bar_width, alpha=opacity, 
                    color=workload_colors[wtype][i], 
                    label=f"{wtype} ({metric.replace('_tps', '')})")
    
    plt.xlabel('Model')
    plt.ylabel('Tokens Per Second (TPS)')
    plt.title('Ollama: Average TPS by Workload Type and Processing Phase')
    plt.xticks(x, model_names, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig("visualizations/ollama/3_model_tps_comparison.png", dpi=300)
    plt.close()
    
    # Plot 4: Input vs Generated Tokens
    plt.figure(figsize=(10, 8))
    for idx, model in enumerate(df_ollama['model'].unique()):
        sub_df = df_ollama[df_ollama['model'] == model]
        plt.scatter(sub_df['input_tokens'], sub_df['generated_tokens'], 
                   color=model_colors[idx], label=sub_df['model_short'].iloc[0], alpha=0.7, edgecolor='k', s=100)
    
    plt.xlabel("Input Tokens")
    plt.ylabel("Generated Tokens")
    plt.title("Ollama: Input vs. Generated Tokens by Model")
    plt.legend()
    plt.tight_layout()
    plt.savefig("visualizations/ollama/4_input_vs_generated_tokens.png", dpi=300)
    plt.close()
    
    # Plot 5: Total processing time by model and dataset
    plt.figure(figsize=(12, 8))
    sns.boxplot(x='model_short', y='total_time_s', hue='dataset', 
               data=df_ollama, palette='Set2')
    plt.xticks(rotation=45)
    plt.xlabel('Model')
    plt.ylabel('Total Processing Time (seconds)')
    plt.title('Ollama: Processing Time by Model and Dataset')
    plt.legend(title='Dataset')
    plt.tight_layout()
    plt.savefig("visualizations/ollama/5_processing_time_by_model_dataset.png", dpi=300)
    plt.close()
    
    # Create an additional plot 6 to match MLX's workload ratio distribution
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    
    # Histogram
    sns.histplot(
        data=df_ollama, 
        x='workload_ratio',
        hue='workload_type',
        bins=20,
        kde=True,
        ax=axes[0],
        palette=COLORS
    )
    axes[0].set_title('Ollama: Distribution of Workload Ratio')
    axes[0].set_xlabel('Workload Ratio (input/total)')
    axes[0].set_ylabel('Count')
    
    # Boxplot by model
    sns.boxplot(
        data=df_ollama, 
        x='model_short', 
        y='workload_ratio',
        hue='workload_type',
        ax=axes[1],
        palette=COLORS
    )
    axes[1].set_title('Ollama: Workload Ratio by Model and Type')
    axes[1].set_xlabel('Model')
    axes[1].set_ylabel('Workload Ratio (input/total)')
    axes[1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig("visualizations/ollama/6_workload_ratio_distribution.png", dpi=300)
    plt.close()
    
    print("✅ Ollama visualizations created successfully.")

####################################
# COMPARISON VISUALIZATIONS
####################################
def create_comparison_visualizations():
    print("Creating comparison visualizations...")
    
    # Check if both files exist
    if not os.path.exists("results/mlx_results.csv") or not os.path.exists("results/ollama_results.csv"):
        print("⚠️ Unable to create comparison visualizations: missing result files")
        return
        
    # Create comparison directory
    os.makedirs("visualizations/comparison", exist_ok=True)
    
    # Load both datasets
    df_mlx = pd.read_csv("results/mlx_results.csv")
    df_ollama = pd.read_csv("results/ollama_results.csv")
    
    # Clean model names for better comparison
    df_mlx['model_short'] = df_mlx['model'].apply(lambda x: x.split('/')[-1].split('-4bit')[0])
    df_ollama['model_short'] = df_ollama['model'].apply(lambda x: x.split(':')[0])
    
    # Add framework column for identification
    df_mlx['framework'] = 'MLX'
    df_ollama['framework'] = 'Ollama'
    
    # Add calculated fields for better comparison
    df_mlx['calculated_time'] = df_mlx['input_tokens'] / df_mlx['prompt_tps'] + df_mlx['generated_tokens'] / df_mlx['generation_tps']
    if 'total_time_s' not in df_mlx.columns:
        df_mlx['total_time_s'] = df_mlx['calculated_time']
    
    # Convert peak_memory to a comparable metric if it doesn't exist in one dataset
    if 'peak_memory' not in df_ollama.columns:
        # Use a placeholder value or create synthetic data
        df_ollama['peak_memory'] = 0  # Placeholder
    
    # Create a mapping between similar models for comparison
    model_map = {
        'TinyLlama-1.1B-Chat-v1.0': 'tinyllama',
        'gemma-2-2b-it': 'gemma',
        'Mistral-7B-Instruct-v0.3': 'mistral'
    }
    
    # Create standardized model names for comparison
    df_mlx['model_standard'] = df_mlx['model_short'].map(lambda x: next((v for k, v in model_map.items() if k in x), x))
    df_ollama['model_standard'] = df_ollama['model_short']
    
    # Select only models that exist in both frameworks
    mlx_models = set(df_mlx['model_standard'].unique())
    ollama_models = set(df_ollama['model_standard'].unique())
    common_models = mlx_models.intersection(ollama_models)
    
    df_mlx_filtered = df_mlx[df_mlx['model_standard'].isin(common_models)]
    df_ollama_filtered = df_ollama[df_ollama['model_standard'].isin(common_models)]
    
    # Combine the dataframes
    df_combined = pd.concat([df_mlx_filtered, df_ollama_filtered], ignore_index=True)
    
    if len(df_combined) == 0:
        print("⚠️ No matching models found between frameworks. Skipping comparison visualizations.")
        return
    
    print(f"Found {len(common_models)} common models for comparison: {', '.join(common_models)}")
    
    # 1. Comparison of Generation TPS across frameworks
    plt.figure(figsize=(14, 8))
    sns.barplot(
        data=df_combined, 
        x='model_standard', 
        y='generation_tps', 
        hue='framework',
        palette=['#3498db', '#e74c3c']
    )
    plt.title('Comparison: Generation Speed by Framework and Model')
    plt.xlabel('Model')
    plt.ylabel('Generation Tokens Per Second')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("visualizations/comparison/1_generation_speed_comparison.png", dpi=300)
    plt.close()
    
    # 2. Comparison of Prompt TPS across frameworks
    plt.figure(figsize=(14, 8))
    sns.barplot(
        data=df_combined, 
        x='model_standard', 
        y='prompt_tps', 
        hue='framework',
        palette=['#3498db', '#e74c3c']
    )
    plt.title('Comparison: Prompt Processing Speed by Framework and Model')
    plt.xlabel('Model')
    plt.ylabel('Prompt Tokens Per Second')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("visualizations/comparison/2_prompt_speed_comparison.png", dpi=300)
    plt.close()
    
    # 3. Comparison of Total Processing Time
    plt.figure(figsize=(14, 8))
    sns.boxplot(
        data=df_combined, 
        x='model_standard', 
        y='total_time_s', 
        hue='framework',
        palette=['#3498db', '#e74c3c']
    )
    plt.title('Comparison: Total Processing Time by Framework and Model')
    plt.xlabel('Model')
    plt.ylabel('Total Processing Time (seconds)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("visualizations/comparison/3_processing_time_comparison.png", dpi=300)
    plt.close()
    
    # 4. Performance by Workload Type across frameworks
    plt.figure(figsize=(14, 8))
    sns.boxplot(
        data=df_combined, 
        x='workload_type', 
        y='generation_tps', 
        hue='framework',
        palette=['#3498db', '#e74c3c']
    )
    plt.title('Comparison: Generation Speed by Workload Type and Framework')
    plt.xlabel('Workload Type')
    plt.ylabel('Generation Tokens Per Second')
    plt.tight_layout()
    plt.savefig("visualizations/comparison/4_workload_performance_comparison.png", dpi=300)
    plt.close()
    
    print("✅ Comparison visualizations created successfully.")

####################################
# MAIN FUNCTION
####################################
def main():
    print("Starting visualization generation...")
    
    # Check if MLX results exist
    if os.path.exists("results/mlx_results.csv"):
        create_mlx_visualizations()
    else:
        print("⚠️ MLX results file not found. Skipping MLX visualizations.")
    
    # Check if Ollama results exist
    if os.path.exists("results/ollama_results.csv"):
        create_ollama_visualizations()
    else:
        print("⚠️ Ollama results file not found. Skipping Ollama visualizations.")
    
    # Create comparison visualizations if both files exist
    if os.path.exists("results/mlx_results.csv") and os.path.exists("results/ollama_results.csv"):
        create_comparison_visualizations()
    else:
        print("⚠️ Missing result files for comparison visualizations.")
    
    print("\n✅ All visualizations created successfully!")
    print("📊 MLX visualizations in: visualizations/mlx/")
    print("📊 Ollama visualizations in: visualizations/ollama/")
    print("📊 Comparison visualizations in: visualizations/comparison/")

if __name__ == "__main__":
    main() 