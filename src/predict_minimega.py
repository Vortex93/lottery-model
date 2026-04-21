from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

CSV_PATH   = "tmp/2026_04_20_xjpot.csv"
LOOKBACK   = 12
BATCH_SIZE = 32
EPOCHS     = 200
LR         = 1e-3

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        header=None,
        names=["date", "year", "draw_nr", "nums", "mb"],
    )

    df["date"] = pd.to_datetime(df["date"])
    nums = df["nums"].str.split("-", expand=True).astype(int)
    nums.columns = ["n1", "n2", "n3", "n4"]

    df = pd.concat([df.drop(columns=["nums"]), nums], axis=1)

    df["date_ordinal"] = df["date"].map(pd.Timestamp.toordinal)

    cols = [
        "date_ordinal",
        "year",
        "draw_nr",
        "n1",
        "n2",
        "n3",
        "n4",
        "mb",
    ]
    return df[cols].astype(float)


def make_sequences(data: np.ndarray, lookback: int) -> tuple[np.ndarray, np.ndarray]:
    X = []
    y = []

    for i in range(lookback, len(data)):
        X.append(data[i - lookback:i])
        y.append(data[i])

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


class LottoModel(nn.Module):
    def __init__(self, input_size: int, proj_size: int = 64, lstm_hidden: int = 64, output_size: int = 8):
        super().__init__()
        self.input_proj = nn.Linear(input_size, proj_size)   # dense
        self.lstm       = nn.LSTM(
            input_size=proj_size,
            hidden_size=lstm_hidden,
            batch_first=True,
        )
        self.head = nn.Sequential(                            # dense
            nn.Linear(lstm_hidden, 64),
            nn.ReLU(),
            nn.Linear(64, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.input_proj(x))
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        return self.head(x)


df = load_data(CSV_PATH)

scaler = MinMaxScaler()
scaled = scaler.fit_transform(df.values)

X, y = make_sequences(scaled, LOOKBACK)

split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
test_ds  = TensorDataset(torch.tensor(X_test), torch.tensor(y_test))

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

model = LottoModel(
    input_size=X.shape[2],
    output_size=y.shape[1],
).to(DEVICE)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

for epoch in range(EPOCHS):
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

    if (epoch + 1) % 20 == 0:
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

# predict next full row from last LOOKBACK rows
last_window = torch.tensor(X[-1:], dtype=torch.float32).to(DEVICE)

with torch.no_grad():
    pred_scaled = model(last_window).cpu().numpy()

pred_row = scaler.inverse_transform(pred_scaled)[0]

date_ordinal = int(round(pred_row[0]))
year         = int(round(pred_row[1]))
draw_nr      = int(round(pred_row[2]))
n1           = int(round(pred_row[3]))
n2           = int(round(pred_row[4]))
n3           = int(round(pred_row[5]))
n4           = int(round(pred_row[6]))
mb           = int(round(pred_row[7]))

# clamp ranges
year    = max(2000, year)
draw_nr = max(1, draw_nr)
main    = sorted(max(1, min(30, x)) for x in [n1, n2, n3, n4])
mb      = max(1, min(15, mb))

pred_date = datetime.fromordinal(max(1, date_ordinal)).strftime("%Y-%m-%d")

prediction = {
    "date": pred_date,
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

print(prediction)
