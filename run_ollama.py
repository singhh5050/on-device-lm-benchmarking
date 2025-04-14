import time
import csv
import os
import ollama
import requests
from datasets import load_dataset
from transformers import AutoTokenizer
from utils.energy_tracking import PowerMonitorContext, cloud_inference_energy_estimate_w_model_attributes

# MODE
MODE = "quant"  # or "prefill"

# Create results directory
os.makedirs("results", exist_ok=True)
OUTPUT_CSV = f"results/ollama_{MODE}_results2.csv"

# Pick models supported by ollama
if MODE == "prefill":
    MODELS = [
        "tinyllama:latest",
        "phi:latest",
        "gemma:2b",
        "mistral:latest",
    ]
elif MODE == "quant":
    MODELS = [
        # "mistral:latest",
        # "mistral:7b-instruct-q5_1",
        "mistral:7b-instruct-q8_0",
    ]
else:
    raise ValueError("MODE must be 'prefill' or 'quant'")


MODEL_TOKENIZERS = {
    "tinyllama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "phi": "microsoft/phi-2",
    "gemma": "google/gemma-2-2b",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.1",
}

DATASET_NAME = "cnn_dailymail"
DATASET_CONFIG = "3.0.0"

with open(OUTPUT_CSV, "w") as f:
    writer = csv.writer(f)
    writer.writerow([
        "model",
        "input_tokens", "generated_tokens", "total_tokens", "workload_type", "workload_ratio",
        "prompt_tps", "generation_tps", "load_time_s", "total_time_s",
        "on_device_energy_J", "on_device_power_W",
        "cloud_energy_J", "cloud_power_W"
    ])

def get_tokenizer(model_name):
    for key in MODEL_TOKENIZERS:
        if model_name.startswith(key):
            return AutoTokenizer.from_pretrained(MODEL_TOKENIZERS[key])
    raise ValueError(f"No tokenizer mapping for model: {model_name}")

def benchmark_model(model, prefill_tokens, decode_tokens, workload):
    print(f"\n=== Running benchmark for {model} ===")
    print(f"📚 Loading sample from {DATASET_NAME}...")
    try:
        dataset = load_dataset(DATASET_NAME, DATASET_CONFIG, split="validation[:1]")
    except ValueError:
        print("Validation split not found, falling back to train split...")
        dataset = load_dataset(DATASET_NAME, DATASET_CONFIG, split="train[:1]")

    text = ""
    for i in range(min(50, len(dataset))):
        sample = dataset[i]
        article_text = sample.get("article") or sample.get("text") or str(sample)
        text += article_text.strip() + "\n\n"

    instruction = ""
    if workload == "prefill":
        instruction = "Summarize this information in a few sentences: "
    elif workload == "decode":
        instruction = "Continue the story, generating as much as you physically can (pages upon pages) - don't stop. Extrapolate: "
    else:
        raise ValueError("Workload must either be 'prefill' or 'decode'")

    tokenizer = get_tokenizer(model)
    MIN_TOKENS_NEEDED = max([512, 1024, 1536, 2048])
    while len(tokenizer.encode(text, add_special_tokens=False)) < MIN_TOKENS_NEEDED:
        text += text

    full_prompt = f"{instruction}\n\n{text}"
    tokenized_prompt = tokenizer.encode(full_prompt, add_special_tokens=False)
    print(f"🧮 Raw prompt token length before truncation: {len(tokenized_prompt)}")
    truncated_prompt_tokens = tokenized_prompt[:prefill_tokens]
    truncated_prompt = tokenizer.decode(truncated_prompt_tokens, skip_special_tokens=True)

    print(f"🧪 Running benchmark")
    try:
        with PowerMonitorContext(mode="mac") as monitor:
            response = ollama.generate(
                model=model,
                prompt=truncated_prompt,
                options={
                    'num_predict': decode_tokens
                }
            )

        metrics = {
            'total_duration': response.get('total_duration', 0),
            'load_duration': response.get('load_duration', 0),
            'prompt_eval_count': response.get('prompt_eval_count', 0),
            'prompt_eval_duration': response.get('prompt_eval_duration', 0),
            'eval_count': response.get('eval_count', 0),
            'eval_duration': response.get('eval_duration', 0),
            'output_text': response.get('response', '')
        }

        input_tokens = len(truncated_prompt_tokens)
        generated_tokens = metrics['eval_count']
        total_tokens = input_tokens + generated_tokens
        workload_ratio = input_tokens / total_tokens if total_tokens > 0 else 0

        prompt_tps = metrics['prompt_eval_count'] / (metrics['prompt_eval_duration'] * 1e-9) if metrics['prompt_eval_duration'] > 0 else 0
        generation_tps = metrics['eval_count'] / (metrics['eval_duration'] * 1e-9) if metrics['eval_duration'] > 0 else 0
        load_time_s = metrics['load_duration'] * 1e-9
        total_time_s = metrics['total_duration'] * 1e-9

        on_device_energy_metrics = monitor.get_final_estimates()
        on_device_energy = on_device_energy_metrics.get("Measured Energy", "n/a")
        on_device_power = on_device_energy_metrics.get("Average Measured Power", "n/a")
        on_device_energy = float(on_device_energy.replace(" J", ""))
        on_device_power = float(on_device_power.replace(" W", ""))

        cloud_energy_metrics = cloud_inference_energy_estimate_w_model_attributes(
            input_tokens=input_tokens,
            output_tokens=generated_tokens,
            inference_wall_time_sec=total_time_s
        )

        cloud_energy = cloud_energy_metrics["total_energy_joules"]
        cloud_power = cloud_energy / total_time_s

        with open(OUTPUT_CSV, "a") as f:
            writer = csv.writer(f)
            writer.writerow([
                model,
                input_tokens, generated_tokens, total_tokens, workload, round(workload_ratio, 3),
                round(prompt_tps, 2), round(generation_tps, 2), 
                round(load_time_s, 3), round(total_time_s, 3),
                round(on_device_energy, 3), round(on_device_power, 3),
                round(cloud_energy, 3), round(cloud_power, 3)
            ])

        print(f"📊 Stats Summary:")
        print(f"📤 Workload: {workload}-heavy")
        print(f"📤 Input tokens: {input_tokens}")
        print(f"📤 Generated tokens: {generated_tokens}")
        print(f"📤 Total tokens: {total_tokens}")
        print(f"📤 Workload ratio: {workload_ratio:.3f}")
        print(f"📤 Prompt TPS: {prompt_tps:.2f}")
        print(f"📤 Generation TPS: {generation_tps:.2f}")
        print(f"📤 Total time: {total_time_s:.3f}s")
        print(f"📤 On Device Energy: {on_device_energy:.3f} J")
        print(f"📤 On Device Power: {on_device_power:.3f} W")
        print(f"📤 Cloud Energy: {cloud_energy:.3f} J")
        print(f"📤 Cloud Power: {cloud_power:.3f} W")
        print(f"📤 Output preview: {metrics['output_text'][:100]}...\n")

    except Exception as e:
        print(f"Error running benchmark for {model}: {e}")

def is_model_available(model_name):
    try:
        response = requests.get("http://localhost:11434/api/tags")
        if response.status_code == 200:
            data = response.json()
            all_models = []

            if 'models' in data:
                all_models = [m['name'] for m in data['models']]
            elif isinstance(data, list):
                all_models = [m['name'] for m in data if 'name' in m]

            for m in all_models:
                if m == model_name or m.startswith(model_name.split(':')[0] + ':'):
                    return True
            return False
        else:
            return False
    except Exception:
        return False

def download_model(model_name):
    try:
        print(f"⬇️ Downloading model: {model_name}...")
        ollama.pull(model_name)
        print(f"✅ Model {model_name} downloaded successfully")
        return True
    except Exception as e:
        print(f"❌ Error downloading model {model_name}: {e}")
        return False

def main():
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

    for model in MODELS:
        if not is_model_available(model):
            print(f"Model {model} not found locally.")
            success = download_model(model)
            if not success:
                print(f"⚠️ Could not download {model}. Skipping.")
                continue
        else:
            print(f"✅ Model {model} is available")

        if MODE == "prefill":
            for prefill in [512, 1024, 1536, 2048]:
                for sample in range(3):
                    benchmark_model(model, prefill, 100, "prefill")
        elif MODE == "quant":
            for sample in range(3):
                benchmark_model(model, 1024, 500, "prefill")

    print(f"\n✅ All {MODE} benchmarks complete! Results saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()