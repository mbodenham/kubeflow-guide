#!/usr/bin/env python3
"""Qwen3.5-0.8B text generation script.

This script loads the Qwen3.5-0.8B model and generates text from a prompt.
The output is saved to a timestamped file in the mounted volume.

Usage:
    python main.py "Your prompt here" --output output.txt
"""

import argparse
import sys
from pathlib import Path
import os

from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser(
        description="Generate text using Qwen3.5-0.8B model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python main.py "Hello, how are you?" --output result.txt
    python main.py "Write a poem about coding" --temperature 0.7 --max_tokens 256
        """
    )
    parser.add_argument("prompt", type=str, help="Input prompt for text generation")
    parser.add_argument("--output", "-o", type=str, default="output.txt",
                        help="Output file path (default: output.txt)")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3.5-0.8B",
                        help="Model name/path (default: Qwen/Qwen3.5-0.8B)")
    parser.add_argument("--max_tokens", type=int, default=512,
                        help="Maximum tokens to generate (default: 512)")
    parser.add_argument("--temperature", type=float, default=0.3,
                        help="Temperature for generation (default: 0.3, lower = more focused)")
    parser.add_argument("--top_p", type=float, default=0.95,
                        help="Top P for generation (default: 0.95)")

    args = parser.parse_args()

    print(f"🚀 Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    )

    print(f"📝 Input prompt: {args.prompt}")

    # Use chat template for Qwen models (improves responses)
    if hasattr(tokenizer, 'apply_chat_template') and tokenizer.chat_template:
        messages = [{"role": "user", "content": args.prompt}]
        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(model.device)
    else:
        inputs = tokenizer(args.prompt, return_tensors="pt").to(model.device)

    print("⏳ Generating response...")
    outputs = model.generate(
        **inputs,
        max_new_tokens=args.max_tokens,
        do_sample=True,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=1.1,
    )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Clean up response (remove chat template if present)
    if hasattr(tokenizer, 'apply_chat_template') and tokenizer.chat_template:
        # Chat template adds the assistant prefix, keep full response
        pass
    elif response.startswith(args.prompt):
        response = response[len(args.prompt):].lstrip()

    # Save to mounted volume path
    output_path = Path(os.path.join("/mnt/data", args.output))
    output_path.write_text(response)
    
    print(f"✅ Output written to: {output_path}")
    print(f"\n📊 Response:\n{response}")


if __name__ == "__main__":
    main()
