import csv

import torch
from torch.utils.data import DataLoader, TensorDataset

from src.models.mac_irl import InvestorIRLModel
from src.training.trainer import train_investor_model


def test_training_writes_loss_and_weight_history(tmp_path):
    features = torch.randn(8, 2, 2)
    contexts = torch.randn(8, 1)
    labels = torch.randint(0, 2, (8,))
    loader = DataLoader(TensorDataset(features, contexts, labels), batch_size=4)
    model = InvestorIRLModel(num_features=2, num_contexts=1)
    checkpoint = tmp_path / "foreign.pt"
    config = {
        "device": "cpu",
        "epochs": 2,
        "learning_rate": 0.01,
        "weight_decay": 0.0,
        "loss": {"lambda_l1": 0.001},
    }

    train_investor_model(
        model,
        loader,
        config,
        checkpoint,
        feature_names=["underwater", "herd"],
        context_names=["kospi_return_1d"],
    )

    with (tmp_path / "foreign_loss_history.csv").open() as f:
        loss_rows = list(csv.DictReader(f))
    with (tmp_path / "foreign_weight_history.csv").open() as f:
        weight_rows = list(csv.DictReader(f))

    assert [int(row["epoch"]) for row in loss_rows] == [1, 2]
    assert len(weight_rows) == 12
    assert {row["parameter"] for row in weight_rows} == {"beta", "B"}
    assert {int(row["epoch"]) for row in weight_rows} == {0, 1, 2}
    assert all(float(row["weight"]) == 0.0 for row in weight_rows if row["epoch"] == "0")
