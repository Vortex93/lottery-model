from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

CSV_PATH = Path("tmp/2026_04_20_xjpot.csv")
ARTIFACT_PATH = Path("tmp/minimega_model.pt")
LOOKBACK = 12
BATCH_SIZE = 32
EPOCHS = 200
LR = 1e-3
FEATURE_COLUMNS = [
    "date_ordinal",
    "year",
    "draw_nr",
    "n1",
    "n2",
    "n3",
    "n4",
    "mb",
]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_data(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        header=None,
        names=["month_day", "year", "draw_nr", "nums", "mb"],
    )
    df["month_day"] = df["month_day"].astype(str).str.strip()
    df["year"] = df["year"].astype(int)
    df["date"] = pd.to_datetime(
        df["month_day"] + ", " + df["year"].astype(str),
        format="%b %d, %Y",
    )

    nums = df["nums"].str.split("-", expand=True).astype(int)
    nums.columns = ["n1", "n2", "n3", "n4"]

    df = pd.concat([df.drop(columns=["month_day", "nums"]), nums], axis=1)
    df["date_ordinal"] = df["date"].map(pd.Timestamp.toordinal)
    df = df.sort_values(["date", "draw_nr"]).reset_index(drop=True)
    return df[FEATURE_COLUMNS].astype(float)


def build_prediction_input(rows: list[str]) -> pd.DataFrame:
    parsed_rows: list[dict[str, float]] = []

    for row in rows:
        date_text, draw_nr_text, nums_text, mb_text = [part.strip() for part in row.split(",")]
        draw_date = pd.Timestamp(date_text)
        numbers = [int(part) for part in nums_text.split("-")]

        if len(numbers) != 4:
            raise ValueError(f"expected 4 main numbers in '{row}'")

        parsed_rows.append(
            {
                "date_ordinal": float(draw_date.toordinal()),
                "year": float(draw_date.year),
                "draw_nr": float(int(draw_nr_text)),
                "n1": float(numbers[0]),
                "n2": float(numbers[1]),
                "n3": float(numbers[2]),
                "n4": float(numbers[3]),
                "mb": float(int(mb_text)),
            }
        )

    return pd.DataFrame(parsed_rows, columns=FEATURE_COLUMNS)


def make_sequences(data: np.ndarray, lookback: int) -> tuple[np.ndarray, np.ndarray]:
    X = []
    y = []

    for i in range(lookback, len(data)):
        X.append(data[i - lookback:i])
        y.append(data[i])

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def serialize_scaler(scaler: MinMaxScaler) -> dict[str, Any]:
    return {
        "feature_range": [float(value) for value in scaler.feature_range],
        "min_": scaler.min_.tolist(),
        "scale_": scaler.scale_.tolist(),
        "data_min_": scaler.data_min_.tolist(),
        "data_max_": scaler.data_max_.tolist(),
        "data_range_": scaler.data_range_.tolist(),
        "n_features_in_": int(scaler.n_features_in_),
        "n_samples_seen_": int(scaler.n_samples_seen_),
    }


def restore_scaler(payload: dict[str, Any]) -> MinMaxScaler:
    scaler = MinMaxScaler(feature_range=tuple(payload["feature_range"]))
    scaler.min_ = np.array(payload["min_"], dtype=np.float64)
    scaler.scale_ = np.array(payload["scale_"], dtype=np.float64)
    scaler.data_min_ = np.array(payload["data_min_"], dtype=np.float64)
    scaler.data_max_ = np.array(payload["data_max_"], dtype=np.float64)
    scaler.data_range_ = np.array(payload["data_range_"], dtype=np.float64)
    scaler.n_features_in_ = int(payload["n_features_in_"])
    scaler.n_samples_seen_ = int(payload["n_samples_seen_"])
    return scaler


class LottoModel(nn.Module):
    def __init__(self, input_size: int, proj_size: int = 64, lstm_hidden: int = 64, output_size: int = 8):
        super().__init__()
        self.input_proj = nn.Linear(input_size, proj_size)
        self.lstm = nn.LSTM(
            input_size=proj_size,
            hidden_size=lstm_hidden,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(lstm_hidden, 64),
            nn.ReLU(),
            nn.Linear(64, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.input_proj(x))
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        return self.head(x)


def create_model(input_size: int, output_size: int) -> LottoModel:
    return LottoModel(input_size=input_size, output_size=output_size).to(DEVICE)


def load_artifact(path: str | Path) -> dict[str, Any]:
    return torch.load(path, map_location=DEVICE)


def clamp_prediction(pred_row: np.ndarray) -> dict[str, Any]:
    date_ordinal = max(1, int(round(pred_row[0])))
    year = max(2000, int(round(pred_row[1])))
    draw_nr = max(1, int(round(pred_row[2])))
    main = sorted(max(1, min(30, int(round(value)))) for value in pred_row[3:7])
    mb = max(1, min(15, int(round(pred_row[7]))))

    return {
        "date": datetime.fromordinal(date_ordinal).strftime("%Y-%m-%d"),
        "year": year,
        "draw_nr": draw_nr,
        "prediction": {
            "n1": main[0],
            "n2": main[1],
            "n3": main[2],
            "n4": main[3],
            "mb": mb,
        },
    }


def train_model(
    csv_path: str | Path = CSV_PATH,
    artifact_path: str | Path = ARTIFACT_PATH,
    lookback: int = LOOKBACK,
    batch_size: int = BATCH_SIZE,
    epochs: int = EPOCHS,
    lr: float = LR,
) -> dict[str, float]:
    df = load_data(csv_path)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df.values)
    X, y = make_sequences(scaled, lookback)

    if len(X) < 2:
        raise ValueError("not enough rows to build train/test sequences")

    split = max(1, int(len(X) * 0.8))
    if split >= len(X):
        split = len(X) - 1

    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    test_ds = TensorDataset(torch.tensor(X_test), torch.tensor(y_test))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    model = create_model(input_size=X.shape[2], output_size=y.shape[1])
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_loss = 0.0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for xb, yb in train_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)

            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * xb.size(0)

        train_loss /= len(train_loader.dataset)

        if (epoch + 1) % 20 == 0 or epoch == 0 or epoch + 1 == epochs:
            print(f"epoch={epoch + 1:03d} train_loss={train_loss:.6f}")

    model.eval()
    test_loss = 0.0

    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)
            pred = model(xb)
            loss = criterion(pred, yb)
            test_loss += loss.item() * xb.size(0)

    test_loss /= len(test_loader.dataset)
    print(f"test_loss={test_loss:.6f}")

    artifact = {
        "lookback": lookback,
        "feature_columns": FEATURE_COLUMNS,
        "input_size": int(X.shape[2]),
        "output_size": int(y.shape[1]),
        "model_state_dict": model.state_dict(),
        "scaler": serialize_scaler(scaler),
    }

    artifact_path = Path(artifact_path)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, artifact_path)
    print(f"saved_artifact={artifact_path}")

    return {
        "train_loss": float(train_loss),
        "test_loss": float(test_loss),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Mini Mega model and save its artifact.")
    parser.add_argument("--csv-path", default=str(CSV_PATH))
    parser.add_argument("--artifact-path", default=str(ARTIFACT_PATH))
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lookback", type=int, default=LOOKBACK)
    parser.add_argument("--lr", type=float, default=LR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_model(
        csv_path=args.csv_path,
        artifact_path=args.artifact_path,
        lookback=args.lookback,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
    )


if __name__ == "__main__":
    main()
