#!/usr/bin/env python3
"""Qwen3.5-35B-A3B-FP8 inference script."""

import argparse
import sys
from pathlib import Path
import os

from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description="Generate text using Qwen3.5-35B-A3B-FP8")
    parser.add_argument("prompt", type=str, help="Input prompt")
    parser.add_argument("--output", "-o", type=str, default="output.txt", help="Output file path")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3.5-35B-A3B-FP8", help="Model name/path")
    parser.add_argument("--max_tokens", type=int, default=512, help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.3, help="Temperature for generation")
    parser.add_argument("--top_p", type=float, default=0.95, help="Top P for generation")
    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    )

    print(f"Input prompt: {args.prompt}")
    
    # For Qwen models, use chat template if available
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

    outputs = model.generate(
        **inputs,
        max_new_tokens=args.max_tokens,
        do_sample=True,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=1.1,
    )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Remove the input prompt/chat template from the response
    if hasattr(tokenizer, 'apply_chat_template') and tokenizer.chat_template:
        # For chat template, just get the assistant response
        pass
    elif response.startswith(args.prompt):
        response = response[len(args.prompt):].lstrip()

    output_path = Path(os.path.join("/mnt/data", args.output))
    output_path.write_text(response)
    print(f"Output written to: {output_path}")
    print(f"Response: {response}")


if __name__ == "__main__":
    main()
