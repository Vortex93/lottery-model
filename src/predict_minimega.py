from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from train_minimega import (
    ARTIFACT_PATH,
    DEVICE,
    build_prediction_input,
    clamp_prediction,
    create_model,
    load_artifact,
    restore_scaler,
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
    parser = argparse.ArgumentParser(description="Predict the next Mini Mega draw from the constant input window.")
    parser.add_argument("--artifact-path", default=str(ARTIFACT_PATH))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifact_path = Path(args.artifact_path)

    if not artifact_path.exists():
        raise FileNotFoundError(
            f"missing model artifact at {artifact_path}; run train_minimega.py first"
        )

    artifact = load_artifact(artifact_path)
    scaler = restore_scaler(artifact["scaler"])
    input_df = build_prediction_input(PREDICTION_INPUT)

    expected_lookback = int(artifact["lookback"])
    if len(input_df) != expected_lookback:
        raise ValueError(
            f"PREDICTION_INPUT must contain {expected_lookback} rows, got {len(input_df)}"
        )

    model = create_model(
        input_size=int(artifact["input_size"]),
        output_size=int(artifact["output_size"]),
    )
    model.load_state_dict(artifact["model_state_dict"])
    model.eval()

    scaled_input = scaler.transform(input_df.values)
    input_tensor = torch.tensor(np.array([scaled_input], dtype=np.float32)).to(DEVICE)

    with torch.no_grad():
        pred_scaled = model(input_tensor).cpu().numpy()

    pred_row = scaler.inverse_transform(pred_scaled)[0]
    print(clamp_prediction(pred_row))


if __name__ == "__main__":
    main()
