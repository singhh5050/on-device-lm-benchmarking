# On-Device LM Benchmarking: MLX vs Ollama

This repository contains benchmark results comparing the performance of language models (LLMs) running locally on Apple Silicon using two frameworks:
- **MLX**: Apple's machine learning framework optimized for Apple Silicon
- **Ollama**: A lightweight framework for running LLMs locally

## Understanding Prefill vs Decode Workloads

Language model inference consists of two distinct phases:

1. **Prefill (Input Processing)**: The initial processing of the prompt/input tokens
2. **Decode (Generation)**: The sequential generation of each output token

These phases have fundamentally different performance characteristics:

- **Prefill-heavy tasks** involve processing large inputs to generate relatively smaller outputs (e.g., summarization)
- **Decode-heavy tasks** involve processing small inputs to generate extensive outputs (e.g., creative writing)

## Performance Analysis

### Prompt TPS vs Generation TPS

![MLX Prompt vs Generation TPS](visualizations/mlx/1_prompt_vs_gen_tps.png)
*MLX: Relationship between prompt processing and generation speeds*

![Ollama Prompt vs Generation TPS](visualizations/ollama/1_prompt_vs_gen_tps.png)
*Ollama: Relationship between prompt processing and generation speeds*

**Key Observations:**
- Both frameworks consistently process prompt tokens faster than they generate new tokens
- For MLX, TinyLlama shows the widest gap between prompt and generation speeds
- Ollama demonstrates remarkably higher prompt processing speeds, especially for TinyLlama
- Mistral shows the smallest difference between prompt and generation speeds in both frameworks

### Model Performance Comparison

![MLX Model TPS Comparison](visualizations/mlx/3_model_tps_comparison.png)
*MLX: Performance across different models and workload types*

![Ollama Model TPS Comparison](visualizations/ollama/3_model_tps_comparison.png)
*Ollama: Performance across different models and workload types*

**Key Observations:**
- Prefill-heavy workloads show similar performance patterns across both frameworks
- Decode-heavy workloads highlight MLX's optimization for generation tasks
- Ollama's performance is more consistent across workload types
- Smaller models (TinyLlama) show the greatest performance difference between frameworks

### Framework Comparison

![Generation Speed Comparison](visualizations/comparison/1_generation_speed_comparison.png)
*Generation speed comparison between MLX and Ollama across models*

![Prompt Processing Speed Comparison](visualizations/comparison/2_prompt_speed_comparison.png)
*Prompt processing speed comparison between MLX and Ollama across models*

![Workload Performance Comparison](visualizations/comparison/4_workload_performance_comparison.png)
*Performance comparison by workload type between frameworks*

**Key Insights:**

1. **Generation Performance**:
   - MLX has significantly better generation performance for TinyLlama (~69 vs ~47 tokens/sec)
   - Gemma and Mistral show similar generation performance across frameworks
   - MLX appears optimized for generation tasks, especially with smaller models

2. **Prompt Processing**:
   - Ollama consistently outperforms MLX in prompt processing speed
   - The difference is most dramatic with TinyLlama, where Ollama processes prompts ~2x faster
   - Mistral shows similar prompt processing performance in both frameworks

3. **Workload Type Performance**:
   - MLX shows higher median performance for decode-heavy tasks
   - Both frameworks perform similarly on prefill-heavy tasks
   - MLX exhibits greater performance variability across both workload types

## Practical Implications

Based on these benchmark results, developers should consider:

1. **Framework Selection Based on Task Type**:
   - For applications requiring extensive text generation (decode-heavy), MLX may offer better performance
   - For applications processing large inputs with shorter outputs (prefill-heavy), Ollama provides comparable performance with better consistency

2. **Model Selection Trade-offs**:
   - TinyLlama shows the greatest performance difference between frameworks
   - Mistral performs similarly on both frameworks but is significantly slower
   - Gemma offers a balanced middle ground with consistent performance

3. **On-Device Optimization Strategies**:
   - Minimize input length where possible to improve overall performance
   - Consider the full end-to-end latency rather than just tokens per second
   - Model size has a significant impact on both frameworks' performance

## Conclusion

The performance characteristics of language models on-device vary significantly based on the framework, model size, and workload type. While tokens per second is a useful comparison metric, real-world performance should consider end-to-end latency, memory usage, and consistency under various workloads.

MLX shows advantages in generation speed, particularly with smaller models, while Ollama offers more consistent performance and superior prompt processing. The choice between frameworks should be guided by the specific requirements of your application.

For full benchmarking details and to run your own comparisons, see the [MLX_README.md](README_MLX.md) and run the benchmarking scripts included in this repository.
