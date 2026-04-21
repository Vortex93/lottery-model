#!/usr/bin/env python3
import argparse
import random
from statistics import mean, median
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Tuple, Set, List

# ------------------------------------------------------------
# Mini Mega Aruba
# - 4 unique main numbers from 1..30
# - 1 Mega Ball from 1..15
# Total unique outcomes: C(30,4) * 15 = 411,075
# Expected hit index (no repeats): (N + 1)/2 ≈ 205,538
# ------------------------------------------------------------

def generate_draw() -> Tuple[Tuple[int, int, int, int], int]:
    main_numbers = tuple(sorted(random.sample(range(1, 31), 4)))
    mega_ball    = random.randint(1, 15)
    return main_numbers, mega_ball

def format_draw(main: Tuple[int, int, int, int], mega: int) -> str:
    return " ".join(f"{n:02d}" for n in main) + f" | MB {mega:02d}"

def simulate_once(target_draw: Tuple[Tuple[int, int, int, int], int],
                  progress_every: int = 0,
                  seed: int | None = None) -> int:
    """
    Keep generating UNIQUE draws (tracked via a set) until we hit target_draw.
    Returns the number of unique draws it took.
    """
    if seed is not None:
        random.seed(seed)

    seen_draws: Set[Tuple[Tuple[int, int, int, int], int]] = set()
    count = 0

    while True:
        draw = generate_draw()
        if draw in seen_draws:
            continue
        seen_draws.add(draw)
        count += 1

        if progress_every and (count % progress_every == 0):
            print(f"[progress] unique_draws={count}")

        if draw == target_draw:
            return count

def percentile(values: List[int], p: float) -> float:
    """
    Simple percentile (0..100) without numpy. Linear interpolation between ranks.
    """
    if not values:
        return float('nan')
    s = sorted(values)
    k = (len(s) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if f == c:
        return float(s[f])
    return s[f] + (s[c] - s[f]) * (k - f)

def parse_target(target: str) -> Tuple[Tuple[int, int, int, int], int]:
    # "05,18,22,23|01"
    main_part, mega_part = target.split("|", 1)
    main_nums = tuple(sorted(int(x) for x in main_part.split(",") if x.strip()))
    mega_num  = int(mega_part.strip())
    if len(main_nums) != 4 or not all(1 <= n <= 30 for n in main_nums):
        raise ValueError("main numbers must be 4 ints in 1..30")
    if not (1 <= mega_num <= 15):
        raise ValueError("mega ball must be in 1..15")
    return (main_nums, mega_num)

def simulate_many(target_draw: Tuple[Tuple[int, int, int, int], int],
                  runs: int,
                  progress_every: int,
                  seed: int | None,
                  workers: int) -> List[int]:
    results: List[int] = []

    if workers <= 1:
        # single-process
        for i in range(1, runs + 1):
            s = None if seed is None else seed + i
            attempts = simulate_once(target_draw, progress_every, s)
            results.append(attempts)
            print(f"run {i:>2}: {attempts:,} unique draws")
        return results

    # multi-process (faster for many runs)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = []
        for i in range(1, runs + 1):
            s = None if seed is None else seed + i
            futures.append(ex.submit(simulate_once, target_draw, progress_every, s))

        for i, fut in enumerate(as_completed(futures), start=1):
            attempts = fut.result()
            results.append(attempts)
            print(f"run {i:>2}: {attempts:,} unique draws")

    return results

def main():
    parser = argparse.ArgumentParser(
        description="Run X experiments to find how many unique Mini Mega draws it takes to hit a target."
    )
    parser.add_argument("--runs", type=int, default=1000,
                        help="How many experiments to run (default: 10)")
    parser.add_argument("--target", type=str, default="05,18,22,23|01",
                        help="Target draw as 'a,b,c,d|m' (default: 05,18,22,23|01)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Base RNG seed for reproducibility (per-run seed = seed + i)")
    parser.add_argument("--progress-every", type=int, default=0,
                        help="Print progress every N unique draws (0 = silent)")
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel workers (default: 1). Use >1 to speed up many runs.")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    target_draw = parse_target(args.target)
    print("target:", format_draw(*target_draw))

    results = simulate_many(
        target_draw=target_draw,
        runs=args.runs,
        progress_every=args.progress_every,
        seed=args.seed,
        workers=args.workers
    )

    if not results:
        return

    avg = mean(results)
    med = median(results)
    p95 = percentile(results, 95)
    p99 = percentile(results, 99)
    mn  = min(results)
    mx  = max(results)

    print("-" * 48)
    print("results:", ", ".join(f"{r:,}" for r in results))
    print(f"avg     : {avg:,.2f}")
    print(f"median  : {med:,.2f}")
    print(f"p95     : {p95:,.2f}")
    print(f"p99     : {p99:,.2f}")
    print(f"min/max : {mn:,} / {mx:,}")
    print("-" * 48)
    print("theory  : total outcomes = 411,075, expected hit ≈ 205,538")

if __name__ == "__main__":
    main()
