#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import random
import os

def generate_draw():
    main_numbers = sorted(random.sample(range(1, 31), 4))
    mega_ball    = random.randint(1, 15)
    return main_numbers, mega_ball

def format_draw(main, mega):
    return " ".join(f"{n:02d}" for n in main) + f" | MB {mega:02d}"

def main():
    parser = argparse.ArgumentParser(description="Generate Mini Mega Aruba lottery draws.")
    parser.add_argument("runs", type=int, help="How many draws to generate")
    parser.add_argument("--seed", type=int, default=None, help="Optional RNG seed")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    # ensure data directory exists
    os.makedirs("data", exist_ok=True)

    output_path = os.path.join("data", "generated.txt")
    with open(output_path, "w") as f:
        for i in range(1, args.runs + 1):
            main_nums, mega = generate_draw()
            line = f"{i:>6}: {format_draw(main_nums, mega)}"
            f.write(line + "\n")

    print(f"Generated {args.runs} draws → {output_path}")

if __name__ == "__main__":
    main()
