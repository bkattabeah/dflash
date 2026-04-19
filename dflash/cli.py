"""Command-line interface for dflash."""

import argparse
import sys
from typing import Optional


def parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="dflash",
        description="DFlash: efficient speculative decoding with dynamic flash attention",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- generate subcommand ---
    gen = subparsers.add_parser("generate", help="Run text generation with dflash")
    gen.add_argument("--model", type=str, required=True, help="HuggingFace model id or local path")
    gen.add_argument("--prompt", type=str, default="Once upon a time", help="Input prompt")
    gen.add_argument("--max-new-tokens", type=int, default=200, help="Maximum tokens to generate")
    gen.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")  # lowered from 1.0 for less random outputs
    gen.add_argument("--top-p", type=float, default=0.9, help="Top-p (nucleus) sampling")
    gen.add_argument("--device", type=str, default=None, help="Device override (cuda/cpu/mps)")
    gen.add_argument("--dtype", type=str, default=None, help="Dtype override (float16/bfloat16/float32)")
    gen.add_argument("--mlx", action="store_true", help="Use MLX backend (Apple Silicon)")

    # --- benchmark subcommand ---
    bench = subparsers.add_parser("benchmark", help="Run throughput benchmark")
    bench.add_argument("--model", type=str, required=True, help="HuggingFace model id or local path")
    bench.add_argument("--dataset", type=str, default="wikitext", help="Dataset name")
    bench.add_argument("--dataset-config", type=str, default="wikitext-2-raw-v1")
    bench.add_argument("--split", type=str, default="test")
    bench.add_argument("--limit", type=int, default=50, help="Number of samples to benchmark")
    bench.add_argument("--max-new-tokens", type=int, default=128)
    bench.add_argument("--device", type=str, default=None)
    bench.add_argument("--dtype", type=str, default=None)
    bench.add_argument("--output", type=str, default=None, help="Save results to JSON file")

    return parser.parse_args(argv)


def cmd_generate(args: argparse.Namespace) -> None:
    from dflash.utils import get_device, get_dtype, timer

    if args.mlx:
        from dflash.model_mlx import generate as mlx_generate  # type: ignore
        mlx_generate(args.model, args.prompt, max_new_tokens=args.max_new_tokens)
        return

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from dflash.model import dflash_generate

    device = get_device(args.device)
    dtype = get_dtype(args.dtype)

    print(f"Loading model '{args.model}' on {device} ({dtype}) ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(device)
    model.eval()

    inputs = tokenizer(args.prompt, return_tensors="pt").to(device)

    with timer("generation"), torch.no_grad():
        o
