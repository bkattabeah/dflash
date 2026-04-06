# DFlash: Block Diffusion for Flash Speculative Decoding
[**Paper**](https://arxiv.org/abs/2602.06036) | [**Blog**](https://z-lab.ai/projects/dflash/) | [**Models**](https://huggingface.co/collections/z-lab/dflash)

**DFlash** is a lightweight **block diffusion** model designed for speculative decoding. It enables efficient and high-quality parallel drafting.
<br>

<div align="center">
  <img src="https://github.com/user-attachments/assets/ffff3d17-96c9-421c-8d26-97127aafdba9" alt="DFlash Architecture" width="100%">
</div>

https://github.com/user-attachments/assets/5b29cabb-eb95-44c9-8ffe-367c0758de8c

<br>

## 📦 Model Support Plan

### ✅ Supported
- **Qwen3.5-4B**: https://huggingface.co/z-lab/Qwen3.5-4B-DFlash
- **Qwen3.5-9B**: https://huggingface.co/z-lab/Qwen3.5-9B-DFlash
- **Qwen3.5-27B**: https://huggingface.co/z-lab/Qwen3.5-27B-DFlash
- **Qwen3.5-35B-A3B**: https://huggingface.co/z-lab/Qwen3.5-35B-A3B-DFlash
- **Qwen3-Coder-Next**: https://huggingface.co/z-lab/Qwen3-Coder-Next-DFlash
- **gpt-oss-20b**: https://huggingface.co/z-lab/gpt-oss-20b-DFlash
- **gpt-oss-120b**: https://huggingface.co/z-lab/gpt-oss-120b-DFlash
- **Qwen3-4B**: https://huggingface.co/z-lab/Qwen3-4B-DFlash-b16  
- **Qwen3-8B**: https://huggingface.co/z-lab/Qwen3-8B-DFlash-b16  
- **Qwen3-Coder-30B-A3B**: https://huggingface.co/z-lab/Qwen3-Coder-30B-A3B-DFlash
- **Llama-3.1-8B-Instruct**: https://huggingface.co/z-lab/LLaMA3.1-8B-Instruct-DFlash-UltraChat

### 🚧 Coming Soon
- **Qwen3.5-122B-A10B**
- **GLM-4.7-Flash**

> 💡 Feel free to open a GitHub issue if you’d like to request support for additional models!  
> We will also open-source the training recipe soon, so you can train your own DFlash draft model to accelerate any LLM.

<br>

## 🚀 Quick Start

### Part 1: Base Installation (Transformers)

```bash
conda create -n dflash python=3.12
conda activate dflash

git clone https://github.com/z-lab/dflash.git
cd dflash

pip install -e .

# Optionally install flash-attn for faster attention.
# If unavailable, falls back to torch.sdpa (slower speedup, same acceptance length).
# pip install flash-attn --no-build-isolation
```

Only Qwen3 and LLaMA-3.1 models support the Transformers backend.

```python
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

model = AutoModel.from_pretrained(
    "z-lab/Qwen3-8B-DFlash-b16", 
    trust_remote_code=True, 
    dtype="auto", 
    device_map="cuda:0"
).eval()

target = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-8B", 
    dtype="auto", 
    device_map="cuda:0"
).eval()

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
prompt = "How many positive whole-number divisors does 196 have?"
messages = [
    {"role": "user", "content": prompt}
]
# Note: this draft model is used for thinking mode disabled
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

generate_ids = model.spec_generate(
    input_ids=model_inputs["input_ids"], 
    max_new_tokens=2048, 
    temperature=0.0, 
    target=target, 
    stop_token_ids=[tokenizer.eos_token_id]
)

print(tokenizer.decode(generate_ids[0], skip_special_tokens=False))
```

### Part 2: SGLang (Optional)

> **Note:** SGLang and vLLM may have conflicting dependencies. Use a separate conda environment for each.

```bash
conda create -n dflash-sglang python=3.12
conda activate dflash-sglang

git clone https://github.com/z-lab/dflash.git
cd dflash

pip install -e ".[sglang]"
```

Launch the server (baseline):
```bash
python -m sglang.launch_server \
    --model-path Qwen/Qwen3-Coder-30B-A3B-Instruct \
    --tp-size 1 --dtype bfloat16 --attention-backend fa3 \
    --mem-fraction-static 0.75 --trust-remote-code
```

Or launch with DFlash speculative decoding:
```bash
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
export SGLANG_ENABLE_SPEC_V2=1
export SGLANG_ENABLE_DFLASH_SPEC_V2=1
export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1

python -m sglang.launch_server \
    --model-path Qwen/Qwen3-Coder-30B-A3B-Instruct \
    --speculative-algorithm DFLASH \
    --speculative-draft-model-path z-lab/Qwen3-Coder-30B-A3B-DFlash \
    --tp-size 1 --dtype bfloat16 --attention-backend fa3 \
    --mem-fraction-static 0.75 --trust-remote-code
```

Then benchmark against the running server:
```bash
python benchmark.py --backend sglang \
    --base-url http://127.0.0.1:30000 \
    --model Qwen/Qwen3-Coder-30B-A3B-Instruct \
    --dataset humaneval \
    --num-prompts 164 \
    --concurrency 32
```

### Part 3: vLLM (Optional)

> **Note:** SGLang and vLLM may have conflicting dependencies. Use a separate conda environment for each.

```bash
conda create -n dflash-vllm python=3.12
conda activate dflash-vllm

git clone https://github.com/z-lab/dflash.git
cd dflash

pip install -e ".[vllm]"
```

Launch the server (baseline):
```bash
vllm serve Qwen/Qwen3.5-27B \
  --attention-backend flash_attn \
  --max-num-batched-tokens 32768
```

Or launch with DFlash speculative decoding:
```bash
vllm serve Qwen/Qwen3.5-27B \
  --speculative-config '{"method": "dflash", "model": "z-lab/Qwen3.5-27B-DFlash", "num_speculative_tokens": 15}' \
  --attention-backend flash_attn \
  --max-num-batched-tokens 32768
```

Then benchmark against the running server:
```bash
python benchmark.py --backend vllm \
    --base-url http://127.0.0.1:8000 \
    --model Qwen/Qwen3.5-27B \
    --dataset humaneval \
    --num-prompts 164 \
    --concurrency 32
```

## 📊 Evaluation
We provide scripts to reproduce the speedup and acceptance length metrics in the paper. The reported results were tested on NVIDIA H200 or B200 GPUs. **Please note that only Qwen3 series and LLaMA-3.1 models support Transformers backend benchmark. For other models please use SGLang to run the benchmarks.**

All benchmarks share the same datasets (gsm8k, math500, humaneval, mbpp, mt-bench). Datasets are automatically downloaded and cached as JSONL in `cache/` on first run.

```bash
# Transformers (Part 1 install only)
BACKEND=transformers bash run_benchmark.sh

# SGLang (requires Part 2 install, server must be running)
BACKEND=sglang MODEL=Qwen/Qwen3.5-9B BASE_URL=http://127.0.0.1:30000 bash run_benchmark.sh

# vLLM (requires Part 3 install, server must be running)
BACKEND=vllm MODEL=Qwen/Qwen3.5-27B BASE_URL=http://127.0.0.1:8000 bash run_benchmark.sh
```


## **Acknowledgement**

Huge thanks to [@dcw02](https://github.com/dcw02), [@gongy](https://github.com/gongy), and the team at [@modal-labs](https://github.com/modal-labs) for their fast, high-quality support in bringing DFlash to SGLang. And huge thanks as well to [@benchislett](https://github.com/benchislett) at NVIDIA for his work in bringing DFlash to vLLM and helping make it available to the broader serving community.

## **Citation**
If you find DFlash useful, please cite our work. To share feedback on DFlash or request new model support, please fill out this form: [DFlash Feedback](https://forms.gle/4YNwfqb4nJdqn6hq9).

```bibtex
@article{chen2026dflash,
  title   = {{DFlash: Block Diffusion for Flash Speculative Decoding}},
  author  = {Chen, Jian and Liang, Yesheng and Liu, Zhijian},
  journal = {arXiv preprint arXiv:2602.06036},
  year    = {2026}
}
```
