import numpy as np

from src.data.splits import build_cpcv, combine_test_folds


def test_cpcv_produces_45_disjoint_train_test_splits():
    config = {
        "cross_validation": {
            "n_folds": 10,
            "n_test_folds": 2,
            "purged_size": 1,
            "embargo_size": 5,
        }
    }
    cv = build_cpcv(config)
    splits = list(cv.split(np.arange(1000).reshape(-1, 1)))

    assert len(splits) == 45
    for train_indices, test_folds in splits:
        test_indices = combine_test_folds(test_folds)
        assert np.intersect1d(train_indices, test_indices).size == 0
        excluded = np.setdiff1d(np.arange(1000), np.union1d(train_indices, test_indices))
        assert len(excluded) > 0
