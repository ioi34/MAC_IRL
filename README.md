# MAC-IRL

PyTorch 기반 단일종목 투자자 행동 reward 모델입니다. 외국인, 기관, 개인을 서로 독립적으로 학습하고 `skfolio`의 Combinatorial Purged Cross-Validation(CPCV)으로 시계열 성능과 보상 가중치 안정성을 평가합니다.

## 설치

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 입력 데이터

기본 입력은 `samsung_macirl_EXTENDED_2021_2025.csv`이며 컬럼명은
`configs/data_extended.yaml`에서 변경할 수 있습니다.

필수 컬럼:

- `date`
- `adjusted_close` 또는 `close`
- `trading_value`
- 투자자별 총매수수량: `{investor}_buy_quantity`
- 투자자별 총매도수량: `{investor}_sell_quantity`
- 투자자별 총매수금액: `{investor}_buy_value`
- 투자자별 총매도금액: `{investor}_sell_value`

`investor`는 `foreign`, `institution`, `retail`입니다. 순매수금액은
`buy_value - sell_value`로 계산하므로 별도 입력하지 않습니다. 모든 수량과 금액은
음수가 아니어야 하며 액면분할 등으로 단위가 바뀐 경우 사전에 보정해야 합니다.

`kospi200`, `usdkrw`는 향후 context 모델을 위한 선택 컬럼이며 현재 모델에는 사용하지 않습니다.

## 실행

```bash
source .venv/bin/activate
python -m scripts.prepare_data
python -m scripts.train
python -m scripts.evaluate
```

`prepare_data.py`는 252거래일의 과거 분포로 다음 날 행동을 4단계로 라벨링하고 최신 유효 973거래일의 비표준화 특징을 저장합니다. `train.py`는 다음 CPCV 설정으로 45개 split을 실행합니다.

```text
n_folds=10
n_test_folds=2
purged_size=1
embargo_size=5
```

각 split에서 scaler는 train index에만 fit되며 test, purge, embargo 관측치는 학습에서 제외됩니다.

## 모델 기본값

- 행동: 과거 252거래일 rolling quartile 기준 `strong_sell=-2`, `weak_sell=-1`, `weak_buy=1`, `strong_buy=2`
- 예측 시점: `S_t`로 `a_{t+1}` 예측
- 보상특징: 평단 대비 손실구간 반응, 군집추종, 20일 모멘텀, 20일 변동성
- reward: 투자자별 독립 fixed linear `R_i(a)=beta_i^T phi_i(a)`
- loss: 투자자별 독립 NLL + L1
- scaling: scikit-learn `StandardScaler`
- metrics: scikit-learn accuracy, macro F1, log loss, confusion matrix

## 결과

`runs/mac_irl_underwater_cpcv/`에 다음 파일이 생성됩니다.

- `cv_metrics.csv`: split·투자자별 성능
- `cv_metrics_summary.csv`: 성능 평균과 표준편차
- `reward_weights.csv`: split·투자자별 보상 가중치
- `reward_weights_summary.csv`: 가중치 평균, 표준편차, 부호 일관성
- `confusion_matrices.json`
- `split_XX/indices.npz`: 실제 train/test/purge·embargo 제외 인덱스
- `split_XX/{investor}_scaler.joblib`: 해당 split·투자자의 train-only scaler
- `split_XX/{investor}.pt`: 독립 모델 checkpoint

보상특징 공식과 데이터 컬럼은 `configs/*.yaml`과 feature registry를 통해 변경할 수 있습니다.

평단가는 투자자별 총매수·총매도 자료로 갱신합니다. 기존 보유량에는 일별
`rho=0.98`을 적용하고, 매도는 기존 평단을 바꾸지 않으며 기존 보유량을 초과해
매도된 날에는 남은 신규 매수분의 VWAP로 평단을 재설정합니다. `rho`는 실제
보유량이 아니라 최근 거래의 유효 기억을 나타내는 proxy입니다.

당일 상태의 가격은 종가 대신 투자자별 총거래 VWAP를 사용합니다.

```text
trade_vwap_t = (buy_value_t + sell_value_t) / (buy_quantity_t + sell_quantity_t)
underwater_gap_t = max((average_cost_t - trade_vwap_t) / average_cost_t, 0)
phi_underwater(s_t, a) = action * underwater_gap_t
```

`underwater` 가중치가 양수면 손실 구간에서 매수를, 음수면 매도를 선호하는
방향입니다. `t`의 거래로 갱신한 상태는 `a_(t+1)`을 설명하므로 미래 거래정보를
사용하지 않습니다.

## 환율·KOSPI 조건부 가중치

Binary 기준선과 컨텍스트 실험은 다음 데이터셋을 공유합니다.

```bash
python -m scripts.prepare_data \
  --data-config configs/data_context.yaml \
  --features-config configs/features_context.yaml

python -m scripts.train \
  --data-config configs/data_context.yaml \
  --features-config configs/features_context.yaml \
  --model-config configs/model.yaml \
  --train-config configs/train.yaml \
  --experiment-config experiments/2026-06-27/configs/context_parkinson_e5_all.yaml
```

조건부 모델은 환율 1일·5일 로그수익률과 KOSPI200 1일·20일 수익률을 사용해
`w_t = beta + B C_t`를 학습합니다. 컨텍스트는 각 CPCV train split에서만
표준화하며 `context_weights.csv`에 `B`를 저장합니다. 기준 보상특징은
`underwater + herd + momentum + Parkinson volatility`입니다.

각 보상특징은 협업 시 충돌을 줄이기 위해 독립 파일이 계산과 행동 변환을 함께 소유합니다.

```text
src/data/average_cost.py       # VWAP, 유효보유량, 평균단가 상태
src/features/underwater.py     # 손실구간의 매수/매도 reward contrast
src/features/herd.py
src/features/momentum.py
src/features/volatility.py
src/features/registry.py  # 등록과 tensor 조립만 담당
```
