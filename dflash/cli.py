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
    gen.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
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
        output_ids = dflash_generate(
            model,
            inputs["input_ids"],
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )

    generated = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print("\n" + "=" * 60)
    print(generated)
    print("=" * 60)


def cmd_benchmark(args: argparse.Namespace) -> None:
    import json
    from dflash.benchmark import load_and_process_dataset, _limit_dataset
    from dflash.utils import get_device, get_dtype

    device = get_device(args.device)
    dtype = get_dtype(args.dtype)

    print(f"Benchmarking '{args.model}' on {device} ({dtype}) ...")

    dataset = load_and_process_dataset(
        args.model,
        dataset_name=args.dataset,
        dataset_config=args.dataset_config,
        split=args.split,
    )
    dataset = _limit_dataset(dataset, args.limit)

    print(f"Dataset ready: {len(dataset)} samples")
    # Detailed benchmark loop lives in benchmark.py; here we just confirm setup.
    results = {"model": args.model, "samples": len(dataset), "device": str(device)}

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.output}")
    else:
        print(results)


def main(argv: Optional[list] = None) -> None:
    args = parse_args(argv)
    if args.command == "generate":
        cmd_generate(args)
    elif args.command == "benchmark":
        cmd_benchmark(args)
    else:
        print(f"Unknown command: {args.command}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
