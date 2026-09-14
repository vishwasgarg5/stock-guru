from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class LSTMConfig:
    lookback: int = 20
    hidden_size: int = 32
    epochs: int = 10
    learning_rate: float = 1e-3
    seed: int = 42


class LSTMChallenger:
    """Optional PyTorch LSTM challenger; never part of the production path by default."""

    def __init__(self, config: LSTMConfig | None = None):
        self.config = config or LSTMConfig()
        if self.config.lookback <= 0:
            raise ValueError("lookback must be positive")
        if self.config.hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if self.config.epochs <= 0:
            raise ValueError("epochs must be positive")
        if self.config.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.config.seed < 0:
            raise ValueError("seed must be non-negative")
        self.model = None
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    def fit(self, sequences: np.ndarray, targets: np.ndarray) -> "LSTMChallenger":
        x = np.asarray(sequences, dtype=np.float32)
        y = np.asarray(targets, dtype=np.float32)
        if x.ndim != 3 or y.ndim != 2:
            raise ValueError("sequences must be [samples, lookback, features] and targets [samples, outputs]")
        if x.shape[0] != y.shape[0] or x.shape[0] == 0 or x.shape[1] == 0 or x.shape[2] == 0 or y.shape[1] == 0:
            raise ValueError("sequences and targets must contain matching non-empty sample dimensions")
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError("sequences and targets must contain only finite values")
        if self.config.lookback != x.shape[1]:
            raise ValueError("sequence lookback does not match LSTMConfig.lookback")
        try:
            import torch
            from torch import nn
        except ImportError as exc:
            raise RuntimeError("PyTorch is required only when evaluating the LSTM challenger") from exc
        torch.manual_seed(self.config.seed)
        self.mean_ = x.reshape(-1, x.shape[-1]).mean(axis=0)
        self.std_ = x.reshape(-1, x.shape[-1]).std(axis=0)
        self.std_[self.std_ == 0] = 1.0
        x = (x - self.mean_) / self.std_
        model = nn.LSTM(input_size=x.shape[-1], hidden_size=self.config.hidden_size, batch_first=True)
        head = nn.Linear(self.config.hidden_size, y.shape[-1])

        class Net(nn.Module):
            def __init__(self, rnn, output):
                super().__init__()
                self.rnn = rnn
                self.output = output

            def forward(self, z):
                h, _ = self.rnn(z)
                return self.output(h[:, -1, :])

        self.model = Net(model, head)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        loss_fn = nn.MSELoss()
        xt = torch.from_numpy(x)
        yt = torch.from_numpy(y)
        self.model.train()
        for _ in range(self.config.epochs):
            opt.zero_grad()
            loss = loss_fn(self.model(xt), yt)
            loss.backward()
            opt.step()
        return self

    def predict(self, sequences: np.ndarray) -> np.ndarray:
        if self.model is None or self.mean_ is None or self.std_ is None:
            raise RuntimeError("Fit the LSTM challenger before prediction")
        import torch
        x = np.asarray(sequences, dtype=np.float32)
        if x.ndim != 3 or x.shape[2] != len(self.mean_):
            raise ValueError("sequences must be [samples, lookback, features] with the fitted feature count")
        if x.shape[1] != self.config.lookback:
            raise ValueError("sequence lookback does not match LSTMConfig.lookback")
        if not np.isfinite(x).all():
            raise ValueError("sequences must contain only finite values")
        x = (x - self.mean_) / self.std_
        self.model.eval()
        with torch.no_grad():
            return self.model(torch.from_numpy(x)).cpu().numpy()


def temporal_challenger_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    """Return dependency-free regression metrics for an evaluated challenger."""
    y = np.asarray(actual, dtype=np.float64)
    yh = np.asarray(predicted, dtype=np.float64)
    if y.ndim != 2 or yh.ndim != 2 or y.shape != yh.shape or y.shape[0] == 0 or y.shape[1] == 0:
        raise ValueError("actual and predicted must be matching non-empty 2D arrays")
    if not np.isfinite(y).all() or not np.isfinite(yh).all():
        raise ValueError("actual and predicted must contain only finite values")
    error = yh - y
    return {
        "samples": int(y.shape[0]),
        "outputs": int(y.shape[1]),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error ** 2))),
    }
