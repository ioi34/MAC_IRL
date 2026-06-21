import numpy as np

from src.features.scaling import fit_feature_scaler


def test_standard_scaler_fit_uses_train_indices_only():
    features = np.ones((4, 1, 2, 1), dtype=np.float32)
    features[2:] = 100.0
    scaler = fit_feature_scaler(features, np.array([0, 1]))

    assert scaler.mean_.tolist() == [1.0]
    assert scaler.scale_.tolist() == [1.0]
