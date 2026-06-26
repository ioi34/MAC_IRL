from __future__ import annotations

import numpy as np
import pandas as pd


# 모델 학습용으로 액션 값을 0부터 시작하는 정수 인덱스로 바꿀 때 쓰는 표입니다.
# 원래 라벨은 -2, -1, 1, 2이고, 학습 배열에서는 각각 0, 1, 2, 3으로 표현합니다.
ACTION_TO_INDEX = {-2: 0, -1: 1, 1: 2, 2: 3}

# 위 표를 반대로 뒤집은 표입니다. 예: 0 -> -2, 3 -> 2
INDEX_TO_ACTION = {v: k for k, v in ACTION_TO_INDEX.items()}


def investor_trade_imbalance(df: pd.DataFrame, config: dict, investor: str) -> pd.Series:
    """투자자 자신의 거래 안에서 매수/매도 쏠림을 계산합니다."""
    investor_columns = config["columns"]["investor_trades"][investor]
    buy_value = pd.to_numeric(df[investor_columns["buy_value"]], errors="raise")
    sell_value = pd.to_numeric(df[investor_columns["sell_value"]], errors="raise")

    # 전체 시장 거래대금이 아니라 해당 투자자의 총거래금액으로 나눕니다.
    # 값의 범위는 대략 -1~1이고, 양수면 매수 우위, 음수면 매도 우위입니다.
    gross_value = (buy_value + sell_value).replace(0, np.nan)
    return (buy_value - sell_value) / gross_value


def rolling_quartile_action_labels(
    flow: pd.Series,
    window: int = 252,
    quantiles: tuple[float, float, float] = (0.25, 0.50, 0.75),
) -> pd.Series:
    """과거 flow 분포와 오늘 flow를 비교해 4단계 action 라벨을 만듭니다.

    flow는 pandas Series입니다. Series는 날짜 같은 index와 값이 함께 있는
    1차원 데이터라고 보면 됩니다.
    """
    lower_q, middle_q, upper_q = quantiles
    if not 0 < lower_q < middle_q < upper_q < 1:
        raise ValueError("quantiles must satisfy 0 < q1 < q2 < q3 < 1")

    # shift(1)은 오늘 값을 제외하고 어제까지의 데이터만 보게 합니다.
    # rolling(...).quantile(...)은 최근 window개 값의 quantile 기준선을 계산합니다.
    lagged = flow.shift(1).rolling(window=window, min_periods=window)
    lower = lagged.quantile(lower_q)
    middle = lagged.quantile(middle_q)
    upper = lagged.quantile(upper_q)

    # 처음 window개 구간은 비교 기준선이 없으므로 NaN으로 시작합니다.
    labels = pd.Series(np.nan, index=flow.index, dtype="float64")

    # 과거 1년 분포 기준으로 strong sell, weak sell, weak buy, strong buy를 나눕니다.
    # mask(condition, value)는 condition이 True인 위치만 value로 바꿉니다.
    labels = labels.mask(flow < lower, -2)
    labels = labels.mask((flow >= lower) & (flow < middle), -1)
    labels = labels.mask((flow >= middle) & (flow < upper), 1)
    labels = labels.mask(flow >= upper, 2)
    return labels


def add_action_labels(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """투자자별 action 라벨 컬럼을 DataFrame에 추가합니다."""
    # 원본 df를 직접 바꾸지 않기 위해 복사본에 새 컬럼을 추가합니다.
    out = df.copy()
    labeling = config["labeling"]
    action_values = list(config["action_values"])
    expected_actions = list(ACTION_TO_INDEX)
    if action_values != expected_actions:
        raise ValueError(f"action_values must be {expected_actions} for quartile labels")

    # config["investors"]에는 라벨을 만들 투자자 이름들이 들어 있습니다.
    for investor in config["investors"]:
        # 라벨 기준 flow는 투자자 자신의 총거래금액 대비 순매수 비율입니다.
        # 예: (foreign_buy_value - foreign_sell_value) / (foreign_buy_value + foreign_sell_value)
        label_flow = investor_trade_imbalance(out, config, investor)
        quantiles = tuple(labeling.get("quantiles", [0.25, 0.50, 0.75]))
        label = rolling_quartile_action_labels(
            label_flow,
            window=labeling["rolling_window"],
            quantiles=quantiles,
        )

        # label_shift는 라벨을 앞으로 당기는 값입니다.
        # 기본값 1이면 오늘 행에 내일의 action 라벨을 붙입니다.
        shifted = label.shift(-int(labeling.get("label_shift", 1)))

        out[f"action_{investor}"] = shifted

        # 모델은 -2/-1/1/2 같은 액션 값보다 0/1/2/3 인덱스를 다루기 쉬워서 함께 저장합니다.
        out[f"action_idx_{investor}"] = shifted.map(ACTION_TO_INDEX)
    return out


def labels_array(df: pd.DataFrame, investors: list[str]) -> np.ndarray:
    """투자자별 action_idx 컬럼들을 numpy 배열로 꺼냅니다."""
    # 예: investors가 ["foreign", "institution"]이면
    # ["action_idx_foreign", "action_idx_institution"] 컬럼을 선택합니다.
    cols = [f"action_idx_{investor}" for investor in investors]

    # 학습 코드에서 바로 쓰기 쉽도록 pandas DataFrame을 int64 numpy 배열로 바꿉니다.
    return df[cols].to_numpy(dtype=np.int64)
