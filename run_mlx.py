import time
import csv
import os
import re
import argparse
import subprocess
from datasets import load_dataset
from mlx_lm import load, generate

# --- CLI Arguments ---
parser = argparse.ArgumentParser(description="Run MLX benchmark on a dataset.")
parser.add_argument("--model", type=str, help="Name of the MLX model")
parser.add_argument("--dataset", type=str, help="HuggingFace dataset name")
parser.add_argument("--subset", type=str, default=None, help="Optional dataset subset/config (e.g. '3.0.0')")
parser.add_argument("--samples", type=int, default=2, help="Number of samples to run")
parser.add_argument("--workload", type=str, choices=["prefill-heavy", "decode-heavy", "balanced"], 
                    help="Workload type to benchmark")
parser.add_argument("--run-all", action="store_true", default=False, 
                    help="Run benchmarks on all predefined models and datasets")
args = parser.parse_args()

# Default models and datasets to use when --run-all is specified
DEFAULT_MODELS = [
    "mlx-community/Mistral-7B-Instruct-v0.3-4bit",
    "mlx-community/gemma-2-2b-it-4bit",
    "mlx-community/TinyLlama-1.1B-Chat-v1.0-4bit",
]

DEFAULT_DATASETS = [
    ("cnn_dailymail", "3.0.0", "prefill-heavy"),
    ("wikitext", "wikitext-103-v1", "decode-heavy"),
]

# Function to verify if a model exists and trigger download if needed
def verify_model(model_name):
    print(f"\nVerifying model: {model_name}")
    try:
        # Use a simple prompt to trigger model download if needed
        subprocess.run([
            "python", "-m", "mlx_lm.generate",
            "--model", model_name,
            "--prompt", "Hello world",
            "--max-tokens", "10",  # Set a small limit just to check download works
        ])
        return True
    except Exception as e:
        print(f"Error loading model {model_name}: {e}")
        return False

# Function to run a single benchmark
def run_benchmark(model_name, dataset_name, subset, num_samples, workload_type):
    global OUTPUT_CSV
    
    # --- Load Model ---
    print(f"\n🔄 Loading model: {model_name}")
    try:
        model, tokenizer = load(model_name)
        print("✅ Model loaded.\n")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False
    
    # --- Load Dataset ---
    print(f"📚 Loading {num_samples} samples from {dataset_name}...")
    try:
        if subset:
            dataset = load_dataset(dataset_name, subset, split=f"validation[:{num_samples*5}]")
        else:
            dataset = load_dataset(dataset_name, split=f"validation[:{num_samples*5}]")
    except ValueError:
        # Fallback to train split if validation doesn't exist
        print("Validation split not found, falling back to train split...")
        try:
            if subset:
                dataset = load_dataset(dataset_name, subset, split=f"train[:{num_samples*5}]")
            else:
                dataset = load_dataset(dataset_name, split=f"train[:{num_samples*5}]")
        except Exception as e:
            print(f"❌ Error loading dataset: {e}")
            return False
    
    # --- Benchmark Each Sample ---
    for i in range(min(num_samples, len(dataset))):
        # For prefill-heavy, combine multiple articles
        if workload_type == "prefill-heavy":
            # Combine articles for a longer context
            combined_text = ""
            for j in range(min(5, len(dataset) - i)):
                idx = i * 5 + j
                if idx < len(dataset):
                    sample = dataset[idx]
                    text = sample.get("article") or sample.get("text") or sample.get("content") or sample.get("context") or str(sample)
                    combined_text += f"\nArticle {j+1}:\n{text}\n"
            prompt_text = combined_text
            instruction = "Summarize all the above articles in less than 150 words. Concision!"
        else:
            # For decode-heavy, use single samples but request longer outputs
            sample = dataset[i]
            prompt_text = sample.get("article") or sample.get("text") or sample.get("content") or sample.get("context") or str(sample)
            instruction = "Write a detailed analysis with multiple paragraphs. Include background information, analysis, and implications."
    
        # Format as conversation
        conversation = [{"role": "user", "content": f"{instruction}\n\n{prompt_text}"}]
        
        # Transform using the chat template
        full_prompt = tokenizer.apply_chat_template(
            conversation=conversation, 
            add_generation_prompt=True
        )
        
        # Generate with verbose output to get timing information
        print(f"\n🧪 Sample {i+1}/{num_samples}")
        
        # Capture stdout to get the verbose output
        with CaptureStdout() as output_capture:
            # Define max_tokens based on workload type
            if workload_type == "prefill-heavy":
                max_tokens = 250  # Shorter for prefill-heavy tasks
            elif workload_type == "decode-heavy":
                max_tokens = 1000  # Longer for decode-heavy tasks
            else:  # balanced
                max_tokens = 500  # Medium length for balanced tasks
                
            # Use verbose=True to get timing information
            output = generate(
                model=model,
                tokenizer=tokenizer,
                prompt=full_prompt,
                max_tokens=max_tokens,  # Override the default of 256
                verbose=True     # Get detailed token and timing information
            )
        
        # Get the captured output
        verbose_output = output_capture.value
        print(verbose_output)  # Print it back to the console
        
        # Extract timing information using regex
        try:
            # Extract prompt tokens and TPS
            prompt_match = re.search(r'Prompt: (\d+) tokens, ([0-9.]+) tokens-per-sec', verbose_output)
            prompt_tokens = int(prompt_match.group(1)) if prompt_match else 0
            prompt_tps = float(prompt_match.group(2)) if prompt_match else 0
            
            # Extract generation tokens and TPS
            gen_match = re.search(r'Generation: (\d+) tokens, ([0-9.]+) tokens-per-sec', verbose_output)
            gen_tokens = int(gen_match.group(1)) if gen_match else 0
            gen_tps = float(gen_match.group(2)) if gen_match else 0
            
            # Extract peak memory
            mem_match = re.search(r'Peak memory: ([0-9.]+) GB', verbose_output)
            peak_memory = float(mem_match.group(1)) if mem_match else 0
            
            # Calculate total tokens
            total_tokens = prompt_tokens + gen_tokens
            
            # Calculate workload ratio (input/total)
            workload_ratio = prompt_tokens / total_tokens if total_tokens > 0 else 0
            
            # --- Logging ---
            with open(OUTPUT_CSV, "a") as f:
                writer = csv.writer(f)
                writer.writerow([
                    dataset_name, subset or "default", model_name, i, workload_type,
                    prompt_tokens, gen_tokens, total_tokens,
                    round(prompt_tps, 2), round(gen_tps, 2), round(workload_ratio, 3),
                    round(peak_memory, 3)
                ])
            
            # Output summary
            print(f"\n📊 Stats Summary:")
            print(f"📤 Input tokens: {prompt_tokens}")
            print(f"📤 Generated tokens: {gen_tokens}")
            print(f"📤 Total tokens: {total_tokens}")
            print(f"📤 Workload ratio (input/total): {workload_ratio:.3f}")
            print(f"📤 Prompt TPS: {prompt_tps:.2f}")
            print(f"📤 Generation TPS: {gen_tps:.2f}")
            print(f"📤 Peak Memory: {peak_memory:.3f} GB")
            print(f"📤 Output preview: {output[:100]}...\n")
        except Exception as e:
            print(f"Error during logging: {e}")
    
    return True

# --- Function to capture stdout ---
from io import StringIO
import sys

class CaptureStdout:
    def __enter__(self):
        self.stdout = StringIO()
        self.old_stdout = sys.stdout
        sys.stdout = self.stdout
        return self
    
    def __exit__(self, *args):
        sys.stdout = self.old_stdout
        self.value = self.stdout.getvalue()

# --- Main function ---
def main():
    global OUTPUT_CSV
    
    # Make sure output directory exists
    OUTPUT_CSV = "results/mlx_results.csv"
    os.makedirs("results", exist_ok=True)
    
    # Create CSV header if it doesn't exist
    if not os.path.exists(OUTPUT_CSV):
        with open(OUTPUT_CSV, "w") as f:
            writer = csv.writer(f)
            writer.writerow(["dataset", "subset", "model", "sample_index", "workload_type",
                "input_tokens", "generated_tokens", "total_tokens",
                "prompt_tps", "generation_tps", "workload_ratio",
                "peak_memory"])
    
    # Run in different modes based on CLI args
    if args.run_all:
        print("Running all predefined benchmarks...")
        
        # Verify all models are available
        print("Verifying models are available...")
        for model in DEFAULT_MODELS:
            verify_model(model)
            
        # Run all combinations
        for model in DEFAULT_MODELS:
            for dataset, subset, workload in DEFAULT_DATASETS:
                print(f"\n=== Running: {model} on {dataset} ({workload}) ===")
                run_benchmark(model, dataset, subset, args.samples, workload)
                
        print("\n✅ All benchmarks complete! Results saved to results/mlx_results.csv")
                
    elif args.model and args.dataset and args.workload:
        # Single benchmark mode
        MODEL_NAME = args.model
        DATASET_NAME = args.dataset
        DATASET_SUBSET = args.subset
        NUM_EXAMPLES = args.samples
        WORKLOAD_TYPE = args.workload
        
        print(f"\n=== Running: {MODEL_NAME} on {DATASET_NAME} ({WORKLOAD_TYPE}) ===")
        run_benchmark(MODEL_NAME, DATASET_NAME, DATASET_SUBSET, NUM_EXAMPLES, WORKLOAD_TYPE)
        
        print("\n✅ Benchmark complete! Results saved to results/mlx_results.csv")
        
    else:
        # No valid arguments provided
        print("Error: You must either specify --run-all or provide all required arguments (--model, --dataset, --workload).")
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()
