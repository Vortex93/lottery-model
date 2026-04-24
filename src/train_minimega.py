from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset

CSV_PATH = Path("tmp/2026_04_20_xjpot.csv")
ARTIFACT_PATH = Path("tmp/minimega_model.pt")

LOOKBACK = 12
BATCH_SIZE = 32
EPOCHS = 200
LR = 1e-3

MAIN_CLASSES = 30
MEGA_CLASSES = 15
MAIN_COLUMNS = ["n1", "n2", "n3", "n4"]
MAIN_PICKS = 4

FEATURE_COLUMNS = [
    "draw_nr_norm",
    "day_of_week_sin",
    "day_of_week_cos",
    "day_of_year_sin",
    "day_of_year_cos",
    "month_sin",
    "month_cos",
    "gap_days_norm",
]

MODEL_CONFIG = {
    "d_model": 128,
    "heads": 4,
    "layers": 3,
    "dropout": 0.10,
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def parse_csv_file(path: str | Path) -> pd.DataFrame:
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
    nums.columns = MAIN_COLUMNS

    df = pd.concat([df[["date", "draw_nr", "mb"]], nums], axis=1)
    return prepare_draw_frame(df)


def parse_draw_rows(rows: list[str]) -> pd.DataFrame:
    parsed_rows: list[dict[str, Any]] = []

    for row in rows:
        date_text, draw_nr_text, nums_text, mb_text = [
            part.strip() for part in row.split(",")
        ]
        numbers = [int(part) for part in nums_text.split("-")]

        if len(numbers) != MAIN_PICKS:
            raise ValueError(f"expected {MAIN_PICKS} main numbers in '{row}'")

        parsed_rows.append(
            {
                "date": pd.Timestamp(date_text),
                "draw_nr": int(draw_nr_text),
                "n1": numbers[0],
                "n2": numbers[1],
                "n3": numbers[2],
                "n4": numbers[3],
                "mb": int(mb_text),
            }
        )

    return prepare_draw_frame(pd.DataFrame(parsed_rows))


def prepare_draw_frame(df: pd.DataFrame) -> pd.DataFrame:
    prepared = df.copy()
    prepared["date"] = pd.to_datetime(prepared["date"])

    for column in ["draw_nr", *MAIN_COLUMNS, "mb"]:
        prepared[column] = prepared[column].astype(int)

    prepared = prepared.drop_duplicates(
        subset=["date", "draw_nr", *MAIN_COLUMNS, "mb"]
    )
    prepared = prepared.sort_values(["date", "draw_nr"]).reset_index(drop=True)

    day_of_week = prepared["date"].dt.dayofweek.astype(float)
    day_of_year = prepared["date"].dt.dayofyear.astype(float)
    month = prepared["date"].dt.month.astype(float)

    gap_days = prepared["date"].diff().dt.days.fillna(0)
    gap_days = gap_days.clip(lower=0, upper=31)

    prepared["draw_nr_norm"] = prepared["draw_nr"] / 10_000.0

    prepared["day_of_week_sin"] = np.sin(2 * np.pi * day_of_week / 7)
    prepared["day_of_week_cos"] = np.cos(2 * np.pi * day_of_week / 7)

    prepared["day_of_year_sin"] = np.sin(2 * np.pi * day_of_year / 366)
    prepared["day_of_year_cos"] = np.cos(2 * np.pi * day_of_year / 366)

    prepared["month_sin"] = np.sin(2 * np.pi * month / 12)
    prepared["month_cos"] = np.cos(2 * np.pi * month / 12)

    prepared["gap_days_norm"] = gap_days / 31.0

    return prepared


def build_prediction_input(rows: list[str]) -> pd.DataFrame:
    return parse_draw_rows(rows)


class LotteryDataset(Dataset):
    def __init__(self, draw_frame: pd.DataFrame, seq_len: int):
        self.main_draws = torch.tensor(
            draw_frame[MAIN_COLUMNS].values,
            dtype=torch.long,
        )
        self.mega_draws = torch.tensor(
            draw_frame["mb"].values,
            dtype=torch.long,
        )
        self.features = torch.tensor(
            draw_frame[FEATURE_COLUMNS].values,
            dtype=torch.float32,
        )
        self.seq_len = seq_len

    def __len__(self) -> int:
        return max(0, len(self.main_draws) - self.seq_len)

    def __getitem__(
        self,
        index: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x_main = self.main_draws[index:index + self.seq_len]
        x_mega = self.mega_draws[index:index + self.seq_len]
        x_features = self.features[index:index + self.seq_len]

        y_main = self.main_draws[index + self.seq_len]
        y_mega = self.mega_draws[index + self.seq_len]

        main_target = torch.zeros(MAIN_CLASSES)
        mega_target = torch.zeros(MEGA_CLASSES)

        main_target[y_main - 1] = 1.0
        mega_target[y_mega - 1] = 1.0

        return x_main, x_mega, x_features, main_target, mega_target


class LotteryModel(nn.Module):
    def __init__(
        self,
        seq_len: int,
        main_classes: int = MAIN_CLASSES,
        mega_classes: int = MEGA_CLASSES,
        d_model: int = 128,
        heads: int = 4,
        layers: int = 3,
        dropout: float = 0.10,
    ):
        super().__init__()

        self.main_embedding = nn.Embedding(main_classes + 1, d_model)
        self.mega_embedding = nn.Embedding(mega_classes + 1, d_model)
        self.feature_projection = nn.Linear(len(FEATURE_COLUMNS), d_model)
        self.position = nn.Parameter(torch.randn(seq_len, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=heads,
            batch_first=True,
            dropout=dropout,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=layers,
        )

        self.main_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, main_classes),
        )

        self.mega_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, mega_classes),
        )

    def forward(
        self,
        x_main: torch.Tensor,
        x_mega: torch.Tensor,
        x_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        main_x = self.main_embedding(x_main).mean(dim=2)
        mega_x = self.mega_embedding(x_mega)
        feature_x = self.feature_projection(x_features)

        x = main_x + mega_x + feature_x
        x = x + self.position.unsqueeze(0)

        x = self.encoder(x)
        x = x[:, -1]

        return self.main_head(x), self.mega_head(x)


def create_model(
    seq_len: int,
    model_config: dict[str, Any] | None = None,
) -> LotteryModel:
    config = MODEL_CONFIG if model_config is None else model_config
    return LotteryModel(seq_len=seq_len, **config).to(DEVICE)


def load_artifact(path: str | Path) -> dict[str, Any]:
    return torch.load(path, map_location=DEVICE)


def derive_next_draw_context(draw_frame: pd.DataFrame) -> dict[str, Any]:
    positive_gaps = draw_frame["date"].diff().dt.days
    positive_gaps = positive_gaps[positive_gaps > 0]

    next_gap_days = (
        int(round(positive_gaps.tail(12).median())) if not positive_gaps.empty else 3
    )
    next_gap_days = max(1, next_gap_days)

    next_date = draw_frame["date"].iloc[-1] + pd.Timedelta(days=next_gap_days)
    next_draw_nr = int(draw_frame["draw_nr"].iloc[-1]) + 1

    return {
        "date": next_date.strftime("%Y-%m-%d"),
        "year": int(next_date.year),
        "draw_nr": next_draw_nr,
    }


def compute_loss(
    main_logits: torch.Tensor,
    mega_logits: torch.Tensor,
    y_main: torch.Tensor,
    y_mega: torch.Tensor,
    criterion_main: nn.Module,
    criterion_mega: nn.Module,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    main_loss = criterion_main(main_logits, y_main)
    mega_loss = criterion_mega(mega_logits, y_mega)
    total_loss = main_loss + mega_loss

    return main_loss, mega_loss, total_loss


def evaluate_model(
    model: LotteryModel,
    loader: DataLoader,
    criterion_main: nn.Module,
    criterion_mega: nn.Module,
) -> tuple[float, float, float]:
    model.eval()

    total_main_loss = 0.0
    total_mega_loss = 0.0
    total_loss = 0.0

    with torch.no_grad():
        for x_main, x_mega, x_features, y_main, y_mega in loader:
            x_main = x_main.to(DEVICE)
            x_mega = x_mega.to(DEVICE)
            x_features = x_features.to(DEVICE)
            y_main = y_main.to(DEVICE)
            y_mega = y_mega.to(DEVICE)

            main_logits, mega_logits = model(x_main, x_mega, x_features)

            main_loss, mega_loss, loss = compute_loss(
                main_logits,
                mega_logits,
                y_main,
                y_mega,
                criterion_main,
                criterion_mega,
            )

            total_main_loss += float(main_loss.item()) * x_main.size(0)
            total_mega_loss += float(mega_loss.item()) * x_main.size(0)
            total_loss += float(loss.item()) * x_main.size(0)

    dataset_size = len(loader.dataset)

    return (
        total_main_loss / dataset_size,
        total_mega_loss / dataset_size,
        total_loss / dataset_size,
    )


def train_model(
    csv_path: str | Path = CSV_PATH,
    artifact_path: str | Path = ARTIFACT_PATH,
    lookback: int = LOOKBACK,
    batch_size: int = BATCH_SIZE,
    epochs: int = EPOCHS,
    lr: float = LR,
) -> dict[str, float]:
    draw_frame = parse_csv_file(csv_path)

    dataset = LotteryDataset(draw_frame, lookback)
    if len(dataset) < 2:
        raise ValueError("not enough rows to build train/test sequences")

    split = max(1, int(len(dataset) * 0.8))
    if split >= len(dataset):
        split = len(dataset) - 1

    train_ds = Subset(dataset, range(0, split))
    test_ds = Subset(dataset, range(split, len(dataset)))

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
    )

    model = create_model(seq_len=lookback)

    criterion_main = nn.BCEWithLogitsLoss()
    criterion_mega = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    train_main_loss = 0.0
    train_mega_loss = 0.0
    train_total_loss = 0.0

    for epoch in range(epochs):
        model.train()

        running_main_loss = 0.0
        running_mega_loss = 0.0
        running_total_loss = 0.0

        for x_main, x_mega, x_features, batch_y_main, batch_y_mega in train_loader:
            x_main = x_main.to(DEVICE)
            x_mega = x_mega.to(DEVICE)
            x_features = x_features.to(DEVICE)
            batch_y_main = batch_y_main.to(DEVICE)
            batch_y_mega = batch_y_mega.to(DEVICE)

            optimizer.zero_grad()

            main_logits, mega_logits = model(x_main, x_mega, x_features)

            main_loss, mega_loss, loss = compute_loss(
                main_logits,
                mega_logits,
                batch_y_main,
                batch_y_mega,
                criterion_main,
                criterion_mega,
            )

            loss.backward()
            optimizer.step()

            running_main_loss += float(main_loss.item()) * x_main.size(0)
            running_mega_loss += float(mega_loss.item()) * x_main.size(0)
            running_total_loss += float(loss.item()) * x_main.size(0)

        dataset_size = len(train_loader.dataset)

        train_main_loss = running_main_loss / dataset_size
        train_mega_loss = running_mega_loss / dataset_size
        train_total_loss = running_total_loss / dataset_size

        if (epoch + 1) % 20 == 0 or epoch == 0 or epoch + 1 == epochs:
            print(
                f"epoch={epoch + 1:03d} "
                f"train_total={train_total_loss:.6f} "
                f"train_main={train_main_loss:.6f} "
                f"train_mega={train_mega_loss:.6f}"
            )

    test_main_loss, test_mega_loss, test_total_loss = evaluate_model(
        model,
        test_loader,
        criterion_main,
        criterion_mega,
    )

    print(
        f"test_total={test_total_loss:.6f} "
        f"test_main={test_main_loss:.6f} "
        f"test_mega={test_mega_loss:.6f}"
    )

    artifact = {
        "artifact_version": 3,
        "lookback": lookback,
        "main_classes": MAIN_CLASSES,
        "mega_classes": MEGA_CLASSES,
        "main_picks": MAIN_PICKS,
        "main_columns": MAIN_COLUMNS,
        "feature_columns": FEATURE_COLUMNS,
        "model_config": MODEL_CONFIG,
        "model_state_dict": model.state_dict(),
    }

    artifact_path = Path(artifact_path)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, artifact_path)

    print(f"saved_artifact={artifact_path}")

    return {
        "train_total_loss": train_total_loss,
        "train_main_loss": train_main_loss,
        "train_mega_loss": train_mega_loss,
        "test_total_loss": test_total_loss,
        "test_main_loss": test_main_loss,
        "test_mega_loss": test_mega_loss,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the Mini Mega classifier and save its artifact."
    )

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
