# Claude Code 프롬프트 — persist 시차 수정 및 재실행

아래 `---` 사이를 그대로 복사해서 Claude Code에 붙여넣으면 된다.

---

## 배경

`src/features/persist.py`의 `build_persist`가 `investor_trade_imbalance(...).shift(1)`을
반환한다. 즉 행 `t`의 persist 피처는 `a_{t-1}`이다. 그런데 타깃은
`configs/data_continuous.yaml`의 `labeling.label_shift: 1`에 의해 `a_{t+1}`이다.
따라서 실제로는 `a_{t+1}`을 `a_{t-1}`로 회귀하는 **AR(2)** 구조이고, 그 사이의 `a_t`는
모델에 전혀 들어가지 않는다.

`docs/persist_spec_and_results.md`는 이 피처를 "어제의 행동 그 자체"이며
"계수가 행동에 대한 문자 그대로의 AR(1)"이라고 명시한다. 즉 **의도는 AR(1)인데 구현이
AR(2)**다. 이번 작업은 명세 변경이 아니라 문서화된 의도와 구현을 일치시키는 수정이다.

정보 누락도 실재한다. `a_t`는 t일 장 마감 후 공표되므로 t+1일 거래 전에 사용 가능하고,
같은 모델의 momentum(`log(P_t/P_{t-20})`)과 underwater는 이미 t일 정보를 쓴다.
persist만 하루 뒤처져 있어 피처 간 시점 정렬도 어긋나 있다.

삼성전자 2022-2025 원계열에서 `corr(a_{t+1}, a_t)`는 `corr(a_{t+1}, a_{t-1})`보다
일관되게 크다 (외국인 0.402 vs 0.308, 기관 0.157 vs 0.112, 개인 0.344 vs 0.246).

## 목표

persist를 `a_t` 기준으로 바꾼 새 run을 만들고, 기존 canonical
(`runs/continuous_reward3_persist_epochs75`)과 대조 가능한 상태로 남긴다.

## 하지 말 것

- 원고(`paper/*.tex`) 수정 금지. 이번 작업 범위가 아니다.
- 기존 `runs/` 디렉터리 삭제·덮어쓰기 금지. 전부 새 경로에 쓴다.
- 기존 `data/processed/dataset_continuous_reward3_persist.npz` 덮어쓰기 금지.
- `herd` 관련 코드 손대지 말 것. 같은 문제가 있지만 canonical이 쓰지 않으므로 별건이다.
- 현대차(005380) 전이 실험은 이 저장소에 데이터도 스크립트도 없다. 시도하지 말고,
  범위 밖임을 최종 보고에 명시할 것.

## 구현 방식 — 하위호환을 유지할 것

`persist`의 기존 동작을 바꾸지 말고 lag를 설정으로 노출한다. 기존 run 재현성이 유지되고
A/B 대조가 config 한 줄로 끝난다.

`src/features/persist.py`:

```python
def build_persist(df, config, investor, action):
    lag = int(config["features"]["params"].get("persist", {}).get("lag", 1))
    return investor_trade_imbalance(df, config, investor).shift(lag) * action
```

`lag` 기본값은 반드시 **1**로 둔다(기존 동작 보존). 새 실험은 `lag: 0`을 쓴다.

새로 만들 config 3개:

1. `configs/features_continuous_reward3_persist_lag0.yaml`
   - `configs/features_continuous_reward3_persist.yaml` 복사
   - `features.params.persist`를 `{lag: 0}`으로
   - `paths.processed_dataset` → `data/processed/dataset_continuous_reward3_persist_lag0.npz`
   - `paths.processed_metadata` → `data/processed/metadata_continuous_reward3_persist_lag0.json`
2. `configs/experiment_continuous_reward3_persist_lag0.yaml`
   - `configs/experiment_continuous_reward3_persist_epochs75.yaml` 복사
   - `experiment.name` → `continuous_reward3_persist_lag0`
   - `experiment.output_dir` → `runs/continuous_reward3_persist_lag0`
   - `investor_overrides`는 **그대로 유지**. 아래 "함정 6" 확인 후에만 변경한다.
3. `experiments/2026-08-05/configs/persist_lag0_validation/` (검증 스위트 전용 config 폴더)
   - `experiments/2026-07-30/configs/persist_epochs75_validation/`를 복사한 뒤
     features/experiment 경로를 위 1·2번으로 치환

## 실행 순서

각 단계가 끝날 때마다 결과를 확인하고 다음으로 넘어갈 것. 실패하면 멈추고 보고할 것.

```bash
source .venv/bin/activate

# 1) 데이터 생성
python -m scripts.prepare_continuous_data \
  --data-config configs/data_continuous.yaml \
  --features-config configs/features_continuous_reward3_persist_lag0.yaml

# 2) CPCV 45 split 학습
python -m scripts.train_continuous \
  --data-config configs/data_continuous.yaml \
  --features-config configs/features_continuous_reward3_persist_lag0.yaml \
  --model-config configs/model.yaml \
  --train-config configs/train.yaml \
  --experiment-config configs/experiment_continuous_reward3_persist_lag0.yaml

# 3) 검증 스위트 (bootstrap 200 / walk-forward / ridge / ablation)
python -m scripts.run_continuous_reward_validation \
  --data-config configs/data_continuous.yaml \
  --features-config configs/features_continuous_reward3_persist_lag0.yaml \
  --model-config configs/model.yaml \
  --train-config configs/train.yaml \
  --experiment-config configs/experiment_continuous_reward3_persist_lag0.yaml \
  --output-root runs/continuous_reward3_persist_lag0_validation \
  --config-dir experiments/2026-08-05/configs/persist_lag0_validation \
  --bootstrap-resamples 200

# 4) 사후 수치 검증
python -m scripts.verify_paper_numbers \
  --run-dir runs/continuous_reward3_persist_lag0 \
  --output-dir experiments/2026-08-05/verify_persist_lag0
```

## 알려진 함정 — 미리 읽고 시작할 것

1. **`--config-dir` 기본값 버그.** `run_continuous_reward_validation.py`의 `--config-dir`
   기본값이 2026-07-28 실험 폴더를 가리킨다. 반드시 위와 같이 명시할 것. 명시하지 않으면
   과거 실험 config를 덮어쓴다. 실행 후 `git status`로 무관한 파일이 변경되지 않았는지 확인.

2. **`analyze_continuous_reward_validation.py`는 마지막 단계에서 `FileNotFoundError`로
   실패한다.** 이 스크립트는 5특징(`relative` 포함) 구성 전용 하드코딩이라 3특징
   구성에는 원래 적용 불가다. 기존 canonical 실행 때도 동일하게 실패했다. **버그로 취급하지
   말고 넘어갈 것.** 그 앞의 13개 분석 파일은 정상 생성된다.

3. **`compare_exact_lasso.py`는 `RUN` 상수를 하드코딩한다**(30행,
   `runs/continuous_reward3_persist_epochs75`). `--run-dir` 인자를 추가하되 기본값은
   기존 값 그대로 두어 하위호환을 유지할 것.

4. **표본 날짜 범위가 하루 당겨진다.** `shift(1)`이 사라지면 선두 NaN이 하나 줄어든다.
   `sample.size: 973`, `selection: latest`가 고정이라 n은 973으로 같지만 시작일이 달라진다.
   이는 **정상**이다. 다만 실험 노트에 새 `date_start`/`date_end`를 기록하고, 기존 run과
   날짜 범위가 다르다는 점을 명시할 것.

5. **CPCV split 인덱스가 재생성된다.** n이 같으므로 인덱스 자체는 동일할 것으로 예상되지만,
   `runs/continuous_reward3_persist_lag0/split_00/indices.npz`와 기존 run의 같은 파일을
   실제로 비교해서 동일 여부를 확인하고 결과를 노트에 적을 것. 다르면 계수 대조가
   일대일이 아니므로 반드시 보고할 것.

6. **수렴 예산이 빠듯하다 — 이번 작업에서 가장 위험한 지점.**
   기존 하이퍼파라미터는 persist 계수가 작던 시절에 튜닝됐다. 수정 후 계수가 2배 이상
   커지는데 optimizer 예산은 그대로다. Adam은 step당 대략 lr만큼 움직이므로 상한이 있다.

   | 유형 | epoch | batch | 학습표본 | step/epoch | 총 step | lr | 이동 상한 | 목표 persist |
   | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
   | 외국인 | 75 | 256 | ~773 | 4 | 300 | 1e-3 | 0.30 | 0.079 |
   | 기관 | 75 | 2048 | ~773 | 1 | 75 | 5e-4 | **0.0375** | **0.0236** |
   | 개인 | 75 | 512 | ~773 | 2 | 150 | 1e-3 | 0.15 | 0.105 |

   기관과 개인은 여유가 크지 않다. **학습 후 반드시 수렴을 확인할 것:**
   - `runs/.../split_00/{investor}_weight_history.csv`에서 마지막 두 epoch의 계수 차이를
     확인. 마지막 epoch 변화량이 계수 절댓값의 1% 이상이면 미수렴으로 판단.
   - `{investor}_loss_history.csv`의 train MSE가 마지막 구간에서 단조 감소하는지 확인.
   - `compare_exact_lasso.py`로 정확 Lasso 해와 대조. **이것이 최종 판정 기준이다.**
     계수 최대 절대 편차가 0.002를 넘으면 미수렴이다.

   미수렴이면 **epoch만** 늘려서(예: 기관 300, 개인 150) 재실행하고, 정확 Lasso와
   일치할 때까지 반복할 것. batch·lr은 건드리지 말 것 — 과거에 HP 전면 통일이 기관을
   발산시킨 이력이 있다(`configs/experiment_continuous_reward3_persist_hpunified.yaml` 주석).
   epoch을 바꿨다면 그 사실과 이유를 실험 노트에 반드시 적을 것.

## 검증 기준 — 아래와 크게 다르면 멈추고 보고

정확 Lasso로 미리 풀어본 예상값이다(45 split 평균). 부호는 반드시 일치해야 하고,
크기는 ±0.005 이내면 정상으로 본다.

| 유형 | β momentum | β persist | β underwater | α KOSPI | α FX |
| --- | ---: | ---: | ---: | ---: | ---: |
| 외국인 | +0.038 | +0.079 | +0.011 | +0.007 | -0.021 |
| 기관 | -0.001 | +0.024 | -0.000 | -0.017 | -0.004 |
| 개인 | -0.028 | +0.105 | +0.018 | +0.029 | +0.030 |

성능(45 split 평균, 테스트 구간):

| 유형 | 방향 정확도 | 상관 | OOS R² |
| --- | ---: | ---: | ---: |
| 외국인 | 0.650 | 0.364 | 0.185 |
| 기관 | 0.556 | 0.165 | +0.023 |
| 개인 | 0.633 | 0.312 | 0.152 |

특히 다음 두 가지를 명시적으로 확인해서 보고할 것.

- **개인 α KOSPI의 부호가 음수에서 양수로 뒤집히는가.** 기존 -0.0225 → 예상 +0.029.
- **기관 OOS R²가 0을 넘는가.** 기존 -0.004 → 예상 +0.023.

이 둘은 원고 §4.4의 서술을 바꾸는 결과이므로 정확히 확인해야 한다.

## 결과 기록

`CLAUDE.md` §5에 따라 `experiments/2026-08-05/HHMM_persist_lag0_재실행.md`를 한국어로
작성할 것. 다음을 반드시 포함한다.

- 변경한 것 / 고정 조건 / 데이터 / 설정 파일 / 결과 폴더 경로
- 새 `date_start`, `date_end`, n
- CPCV split 인덱스가 기존 run과 동일한지 여부
- **가중치 결과 표** — β·α·B 실제 값, CPCV 표준편차, V1 부호 일관성, V2 부트스트랩 CI,
  각 계수의 identify/withhold 판정. 출처 CSV 경로를 함께 적을 것.
- 기존 canonical(`runs/continuous_reward3_persist_epochs75`)과의 대조표
- 수렴 확인 결과(정확 Lasso 최대 편차 포함)
- epoch을 변경했다면 그 사실과 이유
- 현대차 전이는 저장소에 자산이 없어 미실행임을 `미실행`으로 명시

## 마지막에 보고할 것

1. 위 검증 기준 대비 실제값 (표로)
2. 개인 α KOSPI 부호 반전 여부, 기관 OOS R² 부호 여부
3. 수렴 판정 결과와 epoch 변경 여부
4. V2 부트스트랩에서 기관 persist가 0을 배제하는지 — 원고 제목 후보가 여기에 걸려 있다
5. 예상과 어긋난 것 전부. 추측으로 메우지 말고 어긋났다고 그대로 보고할 것

---

## 참고 — 이 프롬프트의 근거

- 시차 문제 실증: `experiments/2026-08-05/1724_persist_lag_and_exact_lasso.md`
- 요약 브리프: `experiments/2026-08-05/persist_lag_brief.pdf`
- persist 명세 이력: `docs/persist_spec_and_results.md`
- 기존 검증 스위트 실행 기록 및 함정: `experiments/2026-07-30/1536_persist_epochs75_검증스위트.md`
- HP 통일 발산 이력: `experiments/2026-07-30/1450_HP통일_수렴검정.md`
