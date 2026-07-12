# MAC-IRL

PyTorch 기반 단일종목 투자자 행동 reward 모델입니다. 외국인, 기관, 개인을 서로 독립적으로 학습하고 `skfolio`의 Combinatorial Purged Cross-Validation(CPCV)으로 시계열 성능과 보상 가중치 안정성을 평가합니다.

## 설치

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 입력 데이터

기본 입력은 `data/raw/samsung_macirl_EXTENDED_2019_2025.csv`이며 컬럼명은
`configs/data_context.yaml`에서 변경할 수 있습니다.

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

`kospi200_return`, `usdkrw`는 최종 컨텍스트 계산에 사용합니다.

## 실행

```bash
source .venv/bin/activate
python -m scripts.prepare_data
python -m scripts.train
python -m scripts.evaluate
```

`prepare_data.py`는 과거 252거래일 중앙값을 기준으로 다음 날 행동을
매도·매수로 라벨링하고 최신 유효 973거래일의 비표준화 특징을 저장합니다.
`train.py`는 다음 CPCV 설정으로 45개 split을 실행합니다.

```text
n_folds=10
n_test_folds=2
purged_size=1
embargo_size=5
```

각 split에서 scaler는 train index에만 fit되며 test, purge, embargo 관측치는 학습에서 제외됩니다.

## 최종 모델

- 행동: 과거 252거래일 rolling median 기준 `sell=-1`, `buy=1`
- 예측 시점: `S_t`로 `a_{t+1}` 예측
- 보상특징: `underwater`, `herd(5)`, 원화 `momentum(20)`, Parkinson `volatility(20)`
- 컨텍스트: `KOSPI200 return(1d)`, 과거 252일 대비 `USD/KRW level z-score`
- reward: 투자자별 독립 contextual linear `R_i(a)=(beta_i+B_i C_t)^T phi_i(a)`
- loss: 투자자별 독립 NLL + L1
- scaling: scikit-learn `StandardScaler`
- metrics: scikit-learn accuracy, macro F1, log loss, confusion matrix

## 결과

`runs/final_reward_context/`에 다음 파일이 생성됩니다.

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

기본 실행은 최종 설정을 사용합니다.

```bash
python -m scripts.prepare_data
python -m scripts.train
python -m scripts.evaluate
```

환율 수준은 현재 `log(USD/KRW)`를 전일까지의 252거래일 평균·표준편차와
비교한 z-score입니다. 따라서 미래값 없이 현재가 과거 1년보다 높은 환율
국면인지 해석할 수 있습니다. 조건부 모델은 `w_t = beta + B C_t`를 학습하며,
컨텍스트는 각 CPCV train split에서만 추가 표준화됩니다.

각 보상특징은 협업 시 충돌을 줄이기 위해 독립 파일이 계산과 행동 변환을 함께 소유합니다.

```text
src/data/average_cost.py       # VWAP, 유효보유량, 평균단가 상태
src/features/underwater.py     # 손실구간의 매수/매도 reward contrast
src/features/herd.py
src/features/momentum.py
src/features/volatility.py
src/features/registry.py  # 등록과 tensor 조립만 담당
```
