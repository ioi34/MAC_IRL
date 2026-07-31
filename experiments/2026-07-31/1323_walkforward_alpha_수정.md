# walk-forward가 α(컨텍스트 주효과)를 추정하지 않던 버그 수정

- 날짜/시간: 2026-07-31 13:23
- 상태: **코드 수정 완료 / 재실행 미실행** ⚠️
  (샌드박스에 torch 설치가 불가능해 학습을 돌리지 못했다. 아래 "실행 방법" 참조.)
- 목적: `2026-07-30/1629_bootstrap_alpha_수정.md`가 진단한 `context_main_effect`
  미전달 버그가 **부트스트랩에만 수정되고 walk-forward에는 적용되지 않았다.**
  그 결과 (1) α에는 V3 증거가 아예 없고, (2) V3의 β 값이 α 없는 축소 사양에서
  나온 값이다. 원고 §4.2의 V3 문장 전체가 여기에 걸린다.
- 변경한 것: **`scripts/train_continuous_walk_forward.py` 한 파일.** 모델·데이터·설정·
  하이퍼파라미터 일체 불변.
- 고정 조건: 3특징(momentum·persist·underwater), 맥락 2개, κ=1, λ=0.005 통일,
  walk-forward 3창(2023/2024/2025), seed 42, n=973.
- 데이터: `data/processed/dataset_continuous_reward3_persist.npz`
- 설정 파일: `configs/data_continuous.yaml`,
  `configs/features_continuous_reward3_persist.yaml`, `configs/model.yaml`,
  `configs/train.yaml`, `configs/experiment_continuous_reward3_persist_epochs75.yaml`
- 결과 폴더 (예정): `runs/continuous_reward3_persist_epochs75_validation/walk_forward_alphafix/`
  - 대조군(구): `runs/continuous_reward3_persist_epochs75_validation/walk_forward/`

## 근본 원인

`scripts/train_continuous_walk_forward.py:168`(수정 전)

```python
model = ContinuousInvestorIRLModel(
    num_features=len(feature_names),
    num_contexts=len(context_names),
    context_mask=context_mask,
)                       # context_main_effect 인자 없음 -> 기본값 False
```

`src/models/continuous.py:47`의 기본값이 `False`이므로, config가
`model.context_main_effect: true`를 지정해도 walk-forward는 **α 파라미터가 아예 없는
모델을 재적합**했다. `config_snapshot.yaml`에는 `context_main_effect: true`가
기록되어 있어 겉보기로는 정상이라 발견이 늦었다.

부트스트랩(`run_continuous_reward_validation.py:334`)과 CPCV 본학습
(`train_continuous.py:319`)은 정상. **모델 생성 지점 3곳을 전수 확인했고 남은
누락은 없다.**

| 생성 지점 | 경로 | α 전달 | 비고 |
| --- | --- | :---: | --- |
| CPCV 본학습 | `scripts/train_continuous.py:319` | ✅ | 애초에 정상 |
| 월별 블록 부트스트랩 | `scripts/run_continuous_reward_validation.py:334` | ✅ | 2026-07-30 수정 |
| walk-forward | `scripts/train_continuous_walk_forward.py:176` | ✅ | **본 수정** |
| (테스트) | `tests/test_continuous_behavior.py` | — | 의도된 인자 조합, 무관 |

ablation·ridge는 `_run_cpcv`가 `scripts.train_continuous`를 서브프로세스로 부르므로
정상 경로를 탄다.

## 수정 내용 (25 insertions, 1 deletion)

`train_continuous.py`의 처리를 그대로 옮겼다.

1. `_context_main_weights_frame`, `summarize_context_main_weights` import 추가
2. `context_main_effect`를 config에서 읽어 모델 생성자에 전달
3. 창별로 `context_main_frames` 수집
4. `context_main_weights.csv` / `context_main_weights_summary.csv` 저장

## 실행 방법

torch가 있는 로컬 `.venv`에서:

```bash
cd MAC_IRL
python -m scripts.train_continuous_walk_forward \
  --data-config configs/data_continuous.yaml \
  --features-config configs/features_continuous_reward3_persist.yaml \
  --model-config configs/model.yaml \
  --train-config configs/train.yaml \
  --experiment-config configs/experiment_continuous_reward3_persist_epochs75.yaml \
  --output-dir runs/continuous_reward3_persist_epochs75_validation/walk_forward_alphafix \
  --test-years 2023 2024 2025
```

3창 × 3유형 = 9 fit. 구 결과를 덮어쓰지 않도록 출력 폴더를 분리했다.

### 실행 후 확인할 것

1. `walk_forward_alphafix/context_main_weights_summary.csv`가 생겼는가
   (= α의 V3가 처음으로 존재)
2. **β의 V3가 바뀌었는가** — 특히 기관 persist. 구 결과는 1/3 창에서 부호 반전이었고,
   이것이 원고 §4.2에서 기관 persist를 강등한 근거였다. 축소 사양의 산물이었다면
   반전이 사라질 수 있다.

```bash
python3 - <<'PY'
import pandas as pd
old = pd.read_csv("runs/continuous_reward3_persist_epochs75_validation/walk_forward/reward_weights_summary.csv")
new = pd.read_csv("runs/continuous_reward3_persist_epochs75_validation/walk_forward_alphafix/reward_weights_summary.csv")
k = ["investor","feature"]
m = old.merge(new, on=k, suffixes=("_old","_new"))
print(m[k+["mean_old","mean_new","direction_consistency_old","direction_consistency_new",
           "sign_reversal_old","sign_reversal_new"]].to_string(index=False))
PY
```

## 가중치 결과

**미실행 — 결과 없음.** 실행 후 아래 표를 채운다. 추측값을 넣지 말 것.

| 대상 | 피처/가중치 | 값 | 변동성/일관성 | 출처 |
| --- | --- | ---: | ---: | --- |
| — | — | 미실행 | 미실행 | — |

- 해석: 미실행.
- 다음 액션:
  1. 위 명령으로 재실행
  2. β의 V3 반전 여부가 바뀌면 **동료에게 즉시 공유** — §4.2를 그 근거로 고쳐 쓰는 중이다
  3. α의 V3가 생기면 `scripts/apply_1col_layout.py`의 `ALPHA_ROWS`에서 V3 열
     `---`를 실제 값으로 교체 (현재 추측 금지 주석이 달려 있다)
  4. 원고 §4.2 / Table 2의 V3 수치 갱신
