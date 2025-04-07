import time
import csv
import os
import ollama  # Fix: Import the entire ollama module, not just generate
import requests
from datasets import load_dataset

# Create results directory
os.makedirs("results", exist_ok=True)
OUTPUT_CSV = "results/ollama_results.csv"

# Define models to test (Ollama model names)
MODELS = [
    "tinyllama:latest",  # Start with smaller models first
    "phi:latest",        # Another small model option
    "gemma:2b",          # Medium sized model
    "mistral:latest",    # Larger model - last option
]

# Define a simple fallback model if others fail
FALLBACK_MODEL = "phi:latest"  # Known to be small and work reliably

# Define datasets and workload types
DATASETS = [
    ("cnn_dailymail", "3.0.0", "prefill-heavy"),
    ("wikitext", "wikitext-103-v1", "decode-heavy"),
]

# Prepare CSV output
with open(OUTPUT_CSV, "w") as f:
    writer = csv.writer(f)
    writer.writerow([
        "dataset", "subset", "model", "sample_index", "workload_type",
        "input_tokens", "generated_tokens", "total_tokens", "workload_ratio",
        "prompt_tps", "generation_tps", "load_time_s", "total_time_s"
    ])

# Main benchmarking function
def benchmark_model(model, dataset_name, subset, workload_type, num_samples=2):
    print(f"\n=== Running: {model} on {dataset_name} ({workload_type}) ===")
    
    # Load dataset
    print(f"📚 Loading {num_samples} samples from {dataset_name}...")
    try:
        if subset:
            dataset = load_dataset(dataset_name, subset, split=f"validation[:{num_samples*5}]")
        else:
            dataset = load_dataset(dataset_name, split=f"validation[:{num_samples*5}]")
    except ValueError:
        # Fallback to train split
        print("Validation split not found, falling back to train split...")
        if subset:
            dataset = load_dataset(dataset_name, subset, split=f"train[:{num_samples*5}]")
        else:
            dataset = load_dataset(dataset_name, split=f"train[:{num_samples*5}]")
    
    # Process each sample
    for i in range(min(num_samples, len(dataset))):
        # Create prompt based on workload type
        if workload_type == "prefill-heavy":
            # Combine multiple articles for a longer context
            combined_text = ""
            for j in range(min(5, len(dataset) - i)):
                idx = i * 5 + j
                if idx < len(dataset):
                    sample = dataset[idx]
                    text = sample.get("article") or sample.get("text") or sample.get("content") or sample.get("context") or str(sample)
                    combined_text += f"\nArticle {j+1}:\n{text}\n"
            prompt_text = combined_text
            instruction = "Summarize all the above articles in 1-2 sentences each."
        else:
            # For decode-heavy, use single samples
            sample = dataset[i]
            prompt_text = sample.get("article") or sample.get("text") or sample.get("content") or sample.get("context") or str(sample)
            instruction = "Write a detailed analysis with multiple paragraphs."
        
        # Format prompt
        full_prompt = f"{instruction}\n\n{prompt_text}"
        
        print(f"\n🧪 Sample {i+1}/{num_samples}")
        
        # Use ollama library to generate response
        try:
            # Make sure we're not streaming and we want the full response
            start_time = time.time()
            
            # Use the generate() function from the ollama library
            response = ollama.generate(
                model=model,
                prompt=full_prompt,
                options={
                    'num_predict': 500 if workload_type == "decode-heavy" else 200
                }
            )
            end_time = time.time()
            
            # Extract metrics from response
            metrics = {
                'total_duration': response.get('total_duration', 0),
                'load_duration': response.get('load_duration', 0),
                'prompt_eval_count': response.get('prompt_eval_count', 0),
                'prompt_eval_duration': response.get('prompt_eval_duration', 0),
                'eval_count': response.get('eval_count', 0),
                'eval_duration': response.get('eval_duration', 0),
                'output_text': response.get('response', '')
            }
            
            # Calculate derived metrics
            input_tokens = metrics['prompt_eval_count']
            generated_tokens = metrics['eval_count']
            total_tokens = input_tokens + generated_tokens
            workload_ratio = input_tokens / total_tokens if total_tokens > 0 else 0
            
            # Convert nanoseconds to seconds for time calculations
            prompt_tps = metrics['prompt_eval_count'] / (metrics['prompt_eval_duration'] * 1e-9) if metrics['prompt_eval_duration'] > 0 else 0
            generation_tps = metrics['eval_count'] / (metrics['eval_duration'] * 1e-9) if metrics['eval_duration'] > 0 else 0
            load_time_s = metrics['load_duration'] * 1e-9
            total_time_s = metrics['total_duration'] * 1e-9
            
            # Save results to CSV
            with open(OUTPUT_CSV, "a") as f:
                writer = csv.writer(f)
                writer.writerow([
                    dataset_name, subset or "default", model, i, workload_type,
                    input_tokens, generated_tokens, total_tokens, round(workload_ratio, 3),
                    round(prompt_tps, 2), round(generation_tps, 2), 
                    round(load_time_s, 3), round(total_time_s, 3)
                ])
            
            # Print summary
            print(f"📊 Stats Summary:")
            print(f"📤 Input tokens: {input_tokens}")
            print(f"📤 Generated tokens: {generated_tokens}")
            print(f"📤 Total tokens: {total_tokens}")
            print(f"📤 Workload ratio: {workload_ratio:.3f}")
            print(f"📤 Prompt TPS: {prompt_tps:.2f}")
            print(f"📤 Generation TPS: {generation_tps:.2f}")
            print(f"📤 Total time: {total_time_s:.3f}s")
            print(f"📤 Output preview: {metrics['output_text'][:100]}...\n")
            
        except Exception as e:
            print(f"Error running benchmark for {model} on sample {i}: {e}")

# Get available models
def get_available_models():
    """Get list of available models"""
    try:
        # Directly use the API endpoint to get models to avoid structure issues
        response = requests.get("http://localhost:11434/api/tags")
        if response.status_code == 200:
            models_data = response.json()
            # Extract model names from the response, handling different possible structures
            if 'models' in models_data:
                return [model['name'] for model in models_data['models']]
            elif 'models' in models_data.get('tags', {}):  # Some versions use 'tags' container
                return [model['name'] for model in models_data['tags']['models']]
            elif isinstance(models_data, list):  # Some versions return a list directly
                return [model['name'] for model in models_data if 'name' in model]
            else:
                print(f"Warning: Unexpected models list format: {models_data}")
                return []
        else:
            print(f"Error getting models list: {response.status_code}")
            return []
    except Exception as e:
        print(f"Error listing models: {str(e)}")
        return []

# Download a model
def download_model(model_name, timeout=600):  # 10-minute timeout
    """Download a model if not already available"""
    try:
        print(f"⬇️ Downloading model: {model_name}...")
        print(f"This may take several minutes. Please be patient.")
        
        # Use a session with timeout
        session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(max_retries=3)
        session.mount('http://', adapter)
        
        # First check if model already exists in a different format
        base_model = model_name.split(':')[0]
        response = session.get("http://localhost:11434/api/tags", timeout=10)
        if response.status_code == 200:
            data = response.json()
            all_models = []
            
            # Extract all model names from various response formats
            if 'models' in data:
                all_models = [m['name'] for m in data['models']]
            elif isinstance(data, list):
                all_models = [m['name'] for m in data if 'name' in m]
            
            # Check if base model exists
            for m in all_models:
                if m.startswith(base_model + ':') or m == base_model:
                    print(f"✅ Found similar model: {m} - using this instead of {model_name}")
                    return True
        
        # If we need to pull the model, set a timeout
        start_time = time.time()
        
        # Start the pull process
        response = ollama.pull(model_name)
        
        # Check if pull was successful
        elapsed_time = time.time() - start_time
        print(f"✅ Successfully downloaded {model_name} in {elapsed_time:.1f} seconds")
        return True
        
    except Exception as e:
        print(f"❌ Error downloading model {model_name}: {e}")
        
        # Check if the model was partially downloaded
        try:
            print("Checking if model was partially downloaded...")
            response = requests.get("http://localhost:11434/api/tags", timeout=10)
            if response.status_code == 200:
                data = response.json()
                all_models = []
                
                # Extract model names
                if 'models' in data:
                    all_models = [m['name'] for m in data['models']]
                elif isinstance(data, list):
                    all_models = [m['name'] for m in data if 'name' in m]
                
                # Check if our model now exists
                if model_name in all_models:
                    print(f"✅ Model {model_name} appears to be downloaded despite error")
                    return True
                
                # Try alternatives
                base_model = model_name.split(':')[0]
                for m in all_models:
                    if m.startswith(base_model + ':') or m == base_model:
                        print(f"✅ Found alternative model: {m} - using this instead of {model_name}")
                        return True
        except Exception as check_err:
            print(f"Error checking model status: {check_err}")
            
        return False

# Run benchmarks
def main():
    # Check if Ollama is running by querying the version endpoint directly
    try:
        response = requests.get("http://localhost:11434/api/version")
        if response.status_code == 200:
            version_info = response.json()
            print(f"✅ Connected to Ollama server (version: {version_info.get('version', 'unknown')})")
        else:
            print("❌ Error connecting to Ollama server")
            print("Please ensure Ollama is installed and running.")
            return
    except Exception as e:
        print(f"❌ Error connecting to Ollama server: {str(e)}")
        print("Please ensure Ollama is installed and running.")
        return
    
    # Get available models
    available_models = get_available_models()
    print(f"📋 Available models: {', '.join(available_models) if available_models else 'None'}")
    
    # For each model
    for model in MODELS:
        # Check if model exists, pull if not
        if model not in available_models:
            success = download_model(model)
            if not success:
                print(f"⚠️ Could not download {model}. Using fallback model {FALLBACK_MODEL}")
                model = FALLBACK_MODEL
        else:
            print(f"✅ Model {model} is already available")
        
        # For each dataset
        for dataset_name, subset, workload_type in DATASETS:
            benchmark_model(model, dataset_name, subset, workload_type)
    
    print("\n✅ All benchmarks complete! Results saved to results/ollama_results.csv")

if __name__ == "__main__":
    main()