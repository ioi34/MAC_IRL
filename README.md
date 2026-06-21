# MAC-IRL

PyTorch 기반 단일종목 투자자 행동 reward 모델입니다. 외국인, 기관, 개인을 서로 독립적으로 학습하고 `skfolio`의 Combinatorial Purged Cross-Validation(CPCV)으로 시계열 성능과 보상 가중치 안정성을 평가합니다.

## 설치

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 입력 데이터

기본 입력은 `data/raw/daily.csv`이며 컬럼명은 `configs/data.yaml`에서 변경할 수 있습니다.

필수 컬럼:

- `date`
- `adjusted_close` 또는 `close`
- `trading_value`
- `foreign_net_buy`
- `institution_net_buy`
- `retail_net_buy`

`kospi200`, `usdkrw`는 향후 context 모델을 위한 선택 컬럼이며 현재 모델에는 사용하지 않습니다.

## 실행

```bash
source .venv/bin/activate
python -m scripts.prepare_data
python -m scripts.train
python -m scripts.evaluate
```

`prepare_data.py`는 252거래일의 과거 분포로 다음 날 행동을 라벨링하고 최신 유효 973거래일의 비표준화 특징을 저장합니다. `train.py`는 다음 CPCV 설정으로 45개 split을 실행합니다.

```text
n_folds=10
n_test_folds=2
purged_size=1
embargo_size=5
```

각 split에서 scaler는 train index에만 fit되며 test, purge, embargo 관측치는 학습에서 제외됩니다.

## 모델 기본값

- 행동: 과거 252거래일 rolling median 기준 `sell=-1`, `buy=1`
- 예측 시점: `S_t`로 `a_{t+1}` 예측
- 보상특징: 손실회피, 군집추종, 20일 모멘텀, 20일 변동성
- reward: 투자자별 독립 fixed linear `R_i(a)=beta_i^T phi_i(a)`
- loss: 투자자별 독립 NLL + L1
- scaling: scikit-learn `StandardScaler`
- metrics: scikit-learn accuracy, macro F1, log loss, confusion matrix

## 결과

`runs/mac_irl_binary_cpcv/`에 다음 파일이 생성됩니다.

- `cv_metrics.csv`: split·투자자별 성능
- `cv_metrics_summary.csv`: 성능 평균과 표준편차
- `reward_weights.csv`: split·투자자별 보상 가중치
- `reward_weights_summary.csv`: 가중치 평균, 표준편차, 부호 일관성
- `confusion_matrices.json`
- `split_XX/indices.npz`: 실제 train/test/purge·embargo 제외 인덱스
- `split_XX/{investor}_scaler.joblib`: 해당 split·투자자의 train-only scaler
- `split_XX/{investor}.pt`: 독립 모델 checkpoint

보상특징 공식과 데이터 컬럼은 `configs/*.yaml`과 feature registry를 통해 변경할 수 있습니다.
