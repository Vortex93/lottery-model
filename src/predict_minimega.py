from __future__ import annotations

import argparse
from itertools import combinations
from math import log
from pathlib import Path

import torch

from train_minimega import (
    ARTIFACT_PATH,
    DEVICE,
    build_prediction_input,
    create_model,
    derive_next_draw_context,
    load_artifact,
    MAIN_COLUMNS,
    MAIN_PICKS,
)

# Keep this window in chronological order: oldest draw first, newest draw last.
PREDICTION_INPUT = [
    "2026-03-10,1851,02-05-19-30,09",
    "2026-03-13,1852,08-14-15-19,11",
    "2026-03-17,1853,07-10-23-30,10",
    "2026-03-20,1854,02-11-16-22,06",
    "2026-03-24,1855,08-11-17-24,09",
    "2026-03-27,1856,04-11-13-18,02",
    "2026-03-31,1857,02-12-18-25,06",
    "2026-04-02,1858,06-14-17-22,04",
    "2026-04-07,1859,01-10-17-25,07",
    "2026-04-10,1860,13-20-23-28,12",
    "2026-04-14,1861,01-10-17-20,03",
    "2026-04-17,1862,07-08-13-14,02",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict the next Mini Mega draw from the constant input window."
    )
    parser.add_argument("--artifact-path", default=str(ARTIFACT_PATH))
    parser.add_argument("--count", type=int, default=1)
    return parser.parse_args()


def build_main_candidates(
    main_probs: torch.Tensor,
    count: int,
) -> list[tuple[tuple[int, ...], float]]:
    ranked_indexes = torch.argsort(main_probs, descending=True)
    pool_size = min(len(ranked_indexes), max(MAIN_PICKS + count * 2, 12))
    number_pool = (ranked_indexes[:pool_size] + 1).tolist()
    candidates: list[tuple[tuple[int, ...], float]] = []

    for numbers in combinations(number_pool, MAIN_PICKS):
        sorted_numbers = tuple(sorted(int(number) for number in numbers))
        score = sum(
            log(float(main_probs[number - 1]) + 1e-12)
            for number in sorted_numbers
        )
        candidates.append((sorted_numbers, score))

    candidates.sort(key=lambda item: item[1], reverse=True)
    return candidates


def rank_candidate_draws(
    main_logits: torch.Tensor,
    mega_logits: torch.Tensor,
    count: int,
) -> list[dict[str, int]]:
    main_probs = torch.sigmoid(main_logits).cpu()[0]
    mega_probs = torch.sigmoid(mega_logits).cpu()[0]

    main_candidates = build_main_candidates(main_probs, count)
    mega_ranked = torch.argsort(mega_probs, descending=True)
    combined_candidates: list[tuple[float, tuple[int, ...], int]] = []

    for main_numbers, main_score in main_candidates[: max(count * 10, 40)]:
        for mega_index in mega_ranked[: max(count * 5, 10)]:
            mega_number = int(mega_index) + 1
            combined_candidates.append(
                (
                    main_score + log(float(mega_probs[int(mega_index)]) + 1e-12),
                    main_numbers,
                    mega_number,
                )
            )

    combined_candidates.sort(key=lambda item: item[0], reverse=True)
    unique_predictions: list[dict[str, int]] = []
    seen: set[tuple[int, ...]] = set()

    for _, main_numbers, mega_number in combined_candidates:
        draw_key = (*main_numbers, mega_number)
        if draw_key in seen:
            continue

        seen.add(draw_key)
        unique_predictions.append(
            {
                "n1": main_numbers[0],
                "n2": main_numbers[1],
                "n3": main_numbers[2],
                "n4": main_numbers[3],
                "mb": mega_number,
            }
        )

        if len(unique_predictions) >= count:
            break

    return unique_predictions


def main() -> None:
    args = parse_args()
    artifact_path = Path(args.artifact_path)

    if args.count < 1:
        raise ValueError("--count must be at least 1")

    if not artifact_path.exists():
        raise FileNotFoundError(
            f"missing model artifact at {artifact_path}; run train_minimega.py first"
        )

    artifact = load_artifact(artifact_path)
    if int(artifact.get("artifact_version", 0)) != 2:
        raise ValueError(
            "artifact was created by the previous model format; run train_minimega.py first"
        )

    input_draws = build_prediction_input(PREDICTION_INPUT)
    expected_lookback = int(artifact["lookback"])

    if len(input_draws) != expected_lookback:
        raise ValueError(
            f"PREDICTION_INPUT must contain {expected_lookback} rows, got {len(input_draws)}"
        )

    input_main = torch.as_tensor(
        input_draws[MAIN_COLUMNS].to_numpy(),
        dtype=torch.long,
    ).unsqueeze(0).to(DEVICE)
    input_mega = torch.as_tensor(
        input_draws["mb"].to_numpy(),
        dtype=torch.long,
    ).unsqueeze(0).to(DEVICE)

    model = create_model(
        seq_len=expected_lookback,
        model_config=artifact["model_config"],
    )
    model.load_state_dict(artifact["model_state_dict"])
    model.eval()

    with torch.no_grad():
        main_logits, mega_logits = model(input_main, input_mega)

    output = derive_next_draw_context(input_draws)
    output["predictions"] = rank_candidate_draws(main_logits, mega_logits, args.count)
    print(output)


if __name__ == "__main__":
    main()
