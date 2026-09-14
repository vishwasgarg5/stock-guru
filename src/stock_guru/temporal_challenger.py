from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class LSTMConfig:
    lookback: int = 20
    hidden_size: int = 32
    epochs: int = 10
    learning_rate: float = 1e-3


class LSTMChallenger:
    """Optional PyTorch LSTM challenger; never part of the production path by default."""

    def __init__(self, config: LSTMConfig | None = None):
        self.config = config or LSTMConfig()
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
        try:
            import torch
            from torch import nn
        except ImportError as exc:
            raise RuntimeError("PyTorch is required only when evaluating the LSTM challenger") from exc
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
        x = (x - self.mean_) / self.std_
        self.model.eval()
        with torch.no_grad():
            return self.model(torch.from_numpy(x)).cpu().numpy()
