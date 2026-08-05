# persist lag0(AR(1) 정합) 재실행 — 시차 버그 수정 및 canonical 대조

- 날짜/시간: 2026-08-05 18:09~18:35
- 목적: `src/features/persist.py`의 `build_persist`가 `investor_trade_imbalance(...).shift(1)`
  로 `a_{t-1}`을 반환하는데, 타깃은 `label_shift: 1`로 `a_{t+1}`이다. 즉 실제 구조는
  `a_{t+1}`을 `a_{t-1}`로 회귀하는 **AR(2)**(사이의 `a_t`를 건너뜀)였고, 이는
  `docs/persist_spec_and_results.md`가 명시한 "어제의 행동 그 자체"·"AR(1)" 의도와
  어긋난다. `a_t`는 t일 장 마감 후 공표되어 t+1일 거래 전 사용 가능하므로 정보 누락도
  실재한다(momentum·underwater는 이미 t일 정보 사용). 명세를 바꾸는 게 아니라
  **문서화된 의도와 구현을 일치**시키는 수정이며, `lag`를 config로 노출해 하위호환을 유지한다.
- 변경한 것:
  - `src/features/persist.py` — `build_persist`에 `features.params.persist.lag`
    파라미터 추가(기본값 **1**, 기존 동작 완전 보존). `lag: 0`이면 `a_t`(문서화된 정의).
  - `scripts/compare_exact_lasso.py` — `--run-dir` 인자 추가(기본값 기존과 동일,
    하위호환). 데이터 경로도 `run_dir/config_snapshot.yaml`의 `paths.processed_dataset`을
    따라가도록 해 run-dir이 바뀌면 대응하는 데이터셋도 자동으로 맞춰지게 했다
    (`--run-dir`만 바꾸고 데이터 경로를 깜빡하면 split 인덱스가 어긋나는 사고를 방지).
  - `configs/features_continuous_reward3_persist_lag0.yaml`,
    `configs/experiment_continuous_reward3_persist_lag0.yaml`,
    `experiments/2026-08-05/configs/persist_lag0_validation/` 신규.
  - **`herd` 관련 코드는 손대지 않았다.** 원고(`paper/*.tex`)도 손대지 않았다.
- 고정 조건: 3특징(momentum·persist·underwater), 맥락 2개, κ=1, λ=0.005 통일,
  CPCV 10/2/purge1/embargo5 = 45 split, seed 42. batch·lr은 canonical(epochs75)과
  동일하게 유지(**건드리지 않음** — 과거 HP 전면 통일이 기관을 발산시킨 이력 때문).
  바뀐 것은 **persist의 lag(1→0)와, 그로 인해 필요해진 epoch 조정뿐**이다.
- 데이터:
  - 신규: `data/processed/dataset_continuous_reward3_persist_lag0.npz`, n=973,
    **2022-01-06 ~ 2025-12-29** — 기존(lag1)과 **날짜 범위·n이 완전히 동일**하다.
    (`eligible_rows=1466`, `rows_excluded_before_sample=745`, 두 데이터셋 동일값.)
  - 기존(lag1, 대조군, 건드리지 않음): `data/processed/dataset_continuous_reward3_persist.npz`
- 설정 파일: `configs/data_continuous.yaml`,
  `configs/features_continuous_reward3_persist_lag0.yaml`, `configs/model.yaml`,
  `configs/train.yaml`, `configs/experiment_continuous_reward3_persist_lag0.yaml`
- 결과 폴더 (전부 신규 경로, 기존 `runs/` 무엇도 덮어쓰지 않음):
  - 본학습: `runs/continuous_reward3_persist_lag0/`
  - 검증 스위트: `runs/continuous_reward3_persist_lag0_validation/`
  - 정확 Lasso 대조: `experiments/2026-08-05/exact_lasso_lag0*/` (아래 "수렴" 절 참조)
  - 사후 검증: `experiments/2026-08-05/verify_persist_lag0/`
  - 비교 기준(대조군, 건드리지 않음): `runs/continuous_reward3_persist_epochs75/`,
    `runs/continuous_reward3_persist_epochs75_validation/`

## 함정 확인 결과 (프롬프트가 미리 경고한 항목)

| 함정 | 예상 | 실제 | 판정 |
| --- | --- | --- | --- |
| 1. `--config-dir` 기본값 버그 | 명시 안 하면 2026-07-28 폴더 덮어씀 | `--config-dir experiments/2026-08-05/configs/persist_lag0_validation`로 명시, 재생성된 `ablation_baseline_features.yaml`에 `persist: {lag: 0}` 정상 반영 확인. `git status`로 무관 파일 변경 없음 확인 | 회피됨 |
| 2. `analyze_continuous_reward_validation.py` 마지막 단계 실패 | `relative` 피처 하드코딩으로 실패 | 동일하게 실패, 그 앞 13개 분석 파일은 정상 생성 | 예상대로, 버그 아님 |
| 3. `compare_exact_lasso.py` RUN 하드코딩 | `--run-dir` 필요 | 추가 완료, 데이터 경로도 `config_snapshot.yaml`에서 자동 추론하도록 함(사양보다 한 단계 더 안전하게 구현 — run-dir만 바꾸고 data 경로를 깜빡하는 사고 방지) | 해결 + 보강 |
| 4. 표본 날짜 범위가 하루 당겨짐 | 선두 NaN 하나 감소로 시작일 이동 | **이동 없음.** n=973, 시작·종료일 완전 동일, `eligible_rows`도 동일(1466). momentum의 20일 lookback이 persist의 lag보다 항상 더 강한 제약이라 persist의 lag 변경이 표본 경계에 영향을 주지 않았다 | **예상과 다름** — 아래 "예상과 어긋난 것" 참조 |
| 5. CPCV split 인덱스 재생성 | 다를 수 있음, 확인 필요 | `runs/continuous_reward3_persist_lag0/split_XX/indices.npz` 45개 전부 `runs/continuous_reward3_persist_epochs75/`와 `train_indices`·`test_indices`·`excluded_indices` **완전 일치**(함정 4의 결과로 당연함). 계수 대조가 split 단위로 일대일 성립 | 확인됨, 문제 없음 |
| 6. 수렴 예산 빠듯함 | 기관·개인 위험 | **개인이 명백히 미수렴, 외국인도 경계선. 기관은 오히려 수렴.** 예상과 위험군이 달랐다 — 아래 "수렴" 절 참조 | 예상과 다름, 실제로 발생 |

## 수렴 확인 — `compare_exact_lasso.py`가 최종 판정 기준

### 1차 시도 (canonical과 동일 epoch: 외국인·기관·개인 전부 75)

45-split 평균(해석 대상 5계수) 대비 정확 Lasso 최대 편차:

| 대상 | 최대 편차 계수 | 값 | 판정(문턱 0.002) |
| --- | --- | ---: | :---: |
| 외국인 | persist | 0.00246 | **미수렴**(경계) |
| 기관 | (전부) | ≤0.00027 | 수렴 |
| 개인 | persist | 0.02149 | **미수렴** |
| 개인 | kospi α | 0.01224 | **미수렴** |
| 개인 | underwater | 0.00616 | **미수렴** |

출처: `experiments/2026-08-05/exact_lasso_lag0/interpreted_coefficients.csv`

### epoch 조정 (batch·lr 불변, "함정 6" 지시 그대로)

기관 행동 계열이 아니라 **외국인·개인**이 미수렴이었다 — 사전 "이동 상한" 표가
예상한 위험군(기관)과 실제로 미수렴한 위험군(외국인·개인)이 달랐다. 반복 조정:

1. 외국인 75→150, 개인 75→150 (기관 75 유지) → 재대조: 외국인 전부 ≤0.00054(수렴).
   개인 kospi α·underwater는 수렴했으나 persist가 0.00219로 근소 초과.
   출처: `experiments/2026-08-05/exact_lasso_lag0_ep150/`
2. 개인만 150→300 (외국인 150, 기관 75 유지) → 재대조: **전원 수렴.**
   출처: `experiments/2026-08-05/exact_lasso_lag0_final/`

### 최종 채택 HP (canonical 대비 epoch만 다름, batch·lr 전부 canonical과 동일)

| 대상 | epochs (canonical→lag0) | batch | lr |
| --- | :---: | ---: | ---: |
| 외국인 | 75 → **150** | 256 (불변) | 0.001 (불변) |
| 기관 | 75 → 75 (불변) | 2048 (불변) | 0.0005 (불변) |
| 개인 | 75 → **300** | 512 (불변) | 0.001 (불변) |

### 최종 수렴 확인 (해석 대상 5계수, 45-split 평균 vs 정확 Lasso)

| 대상 | 최대 절대편차 | 판정 |
| --- | ---: | :---: |
| 외국인 | 0.00054 (persist) | 수렴 |
| 기관 | 0.00027 (persist) | 수렴 |
| 개인 | 0.00025 (kospi α) | 수렴 |

전부 0.002 문턱 이내. `weight_history.csv` 마지막-epoch 상대변화도 전부 1% 미만
(외국인 0.50%, 기관 0.08%, 개인 0.24%, split_00 기준 persist β). 출처:
`experiments/2026-08-05/exact_lasso_lag0_final/interpreted_coefficients.csv`,
`runs/continuous_reward3_persist_lag0/split_00/{investor}_weight_history.csv`

## 검증 기준 대조 — 전부 통과

정확 Lasso로 사전 계산된 예상값(45 split 평균) 대비, 부호 전부 일치·크기 전부
±0.005 이내(대부분 ±0.001 이내).

### β·α

| 대상 | 계수 | 예상 | 실제 | 편차 |
| --- | --- | ---: | ---: | ---: |
| 외국인 | β momentum | +0.038 | +0.0386 | +0.0006 |
| 외국인 | β persist | +0.079 | +0.0799 | +0.0009 |
| 외국인 | β underwater | +0.011 | +0.0112 | +0.0002 |
| 외국인 | α KOSPI | +0.007 | +0.0063 | −0.0007 |
| 외국인 | α FX | −0.021 | −0.0210 | 0.0000 |
| 기관 | β momentum | −0.001 | −0.00087 | +0.0001 |
| 기관 | β persist | +0.024 | +0.0232 | −0.0008 |
| 기관 | β underwater | −0.000 | −0.00025 | −0.0003 |
| 기관 | α KOSPI | −0.017 | −0.0165 | +0.0005 |
| 기관 | α FX | −0.004 | −0.0036 | +0.0004 |
| 개인 | β momentum | −0.028 | −0.0277 | +0.0003 |
| 개인 | β persist | +0.105 | +0.1050 | 0.0000 |
| 개인 | β underwater | +0.018 | +0.0183 | +0.0003 |
| 개인 | α KOSPI | +0.029 | +0.0289 | −0.0001 |
| 개인 | α FX | +0.030 | +0.0305 | +0.0005 |

출처: `runs/continuous_reward3_persist_lag0/reward_weights_summary.csv`,
`runs/continuous_reward3_persist_lag0/context_main_weights_summary.csv`

### 성능 (45 split 평균)

| 대상 | 방향정확도 예상/실제 | 상관 예상/실제 | OOS R² 예상/실제 |
| --- | --- | --- | --- |
| 외국인 | 0.650 / 0.6486 | 0.364 / 0.3639 | 0.185 / 0.1849 |
| 기관 | 0.556 / 0.5569 | 0.165 / 0.1653 | +0.023 / **+0.0238** |
| 개인 | 0.633 / 0.6329 | 0.312 / 0.3123 | 0.152 / 0.1520 |

출처: `runs/continuous_reward3_persist_lag0/cv_metrics_summary.csv`,
`experiments/2026-08-05/verify_persist_lag0/out_of_sample_r2_summary.csv`

## 기존 canonical(lag1, epochs75) 대조표

| 대상 | 계수 | 구(lag1) | 신(lag0) | 비고 |
| --- | --- | ---: | ---: | --- |
| 외국인 | β persist | +0.0505 | +0.0799 | +58% |
| 기관 | β persist | +0.0102 | +0.0232 | +128% |
| 개인 | β persist | +0.0462 | +0.1050 | +127% |
| 개인 | α KOSPI | **−0.0216** | **+0.0289** | **부호 반전** |
| 기관 | OOS R²(학습평균 기준) | **−0.0033** | **+0.0238** | **부호 반전(음→양)** |
| 기관 | 상관 | 0.0676 | 0.1653 | +144% |
| 개인 | 상관 | 0.2419 | 0.3123 | +29% |

(참고: 프롬프트가 인용한 "개인 α KOSPI 기존 −0.0225"는 제가 `runs/continuous_reward3_persist_epochs75/context_main_weights_summary.csv`에서 직접 읽은 값 −0.02157과 소수 3자리에서 차이가 있다. 어느 시점 재실행분을 인용했는지 추적하지 못했다 — 방향·크기 결론에는 영향 없다.)

## V2(부트스트랩)·V3(walk-forward) — 기관 persist

| 검증 | 구(lag1) | 신(lag0) |
| --- | --- | --- |
| V2 부트스트랩 CI | [+0.0001, +0.0208], 배제(경계 근접) | **[+0.0137, +0.0283], 배제(하한이 0에서 더 멀어짐)** |
| V3 walk-forward 부호반전 | **있음**(1/3 창 음전환) | **없음**(3/3 창 전부 양, +0.0240~+0.0268) |

출처: `runs/continuous_reward3_persist_lag0_validation/weight_bootstrap/bootstrap_reward_weights_summary.csv`,
`runs/continuous_reward3_persist_lag0_validation/analysis/walk_forward_reward_stability.csv`

**기관 persist는 V1(45split 100%)·V2(부트스트랩 CI 배제)·V3(walk-forward 반전 없음)
전부 통과로 격상됐다.** 구 lag1에서 유일한 약점이던 V3 반전이 사라졌다 — 시차
버그가 그 반전 자체의 원인이었을 가능성을 시사한다(원인을 직접 분해하진 않았고
상관관계로만 확인).

## 부수 발견 — 요청받지 않았지만 중요

**`remove_persist` ablation 유의 저하가 외국인 하나에서 세 투자자 전부로 확장됐다.**
구 lag1 검증에서는 persist 제거가 유의하게 나빠지는 게 외국인뿐이었는데(RMSE·상관),
신 lag0에서는 세 투자자 전원의 RMSE·상관이 유의하게 저하된다(기관 상관은
0.153→0.005로 사실상 소멸). q-value 전부 <0.02. 출처:
`runs/continuous_reward3_persist_lag0_validation/analysis/ablation_paired_bootstrap.csv`

## 예상과 어긋난 것

1. **날짜 범위가 이동하지 않았다.** "함정 4"는 시작일이 하루 당겨질 것으로 예상했지만
   실제로는 n=973, 시작·종료일, `eligible_rows`가 구·신 완전히 동일했다. 원인:
   momentum의 20일 lookback이 persist의 lag(1 또는 0)보다 항상 더 긴 선행 결측을
   요구하므로, persist 쪽 결측이 한 칸 줄어도 표본 경계를 결정하는 것은 여전히
   momentum이다.
2. **미수렴 위험군이 예상과 달랐다.** 사전 "이동 상한" 표는 기관을 가장 위험하다고
   지목했지만(여유 배수 1.6배), 실제로 기관은 epoch75 그대로 수렴했고 개인(여유
   배수 1.4배와 비슷했음에도 명백히 미수렴, 최종 300epoch 필요)과 외국인(여유가
   가장 컸음에도 경계선)이 문제였다. "이동 상한" 어림값은 실제 수렴 여부를 잘
   예측하지 못했다 — `compare_exact_lasso.py` 없이 사전 추정만 믿었다면 잘못된
   투자자에 예산을 배정했을 것이다.
3. **α KOSPI 개인 기준값의 소수 3자리 불일치** (위 대조표 참고, 결론에 영향 없음).

## 범위 밖 — 미실행

**현대차(005380) 전이 실험은 이 저장소에 데이터도 스크립트도 없어 시도하지
않았다.** 프롬프트가 명시한 대로 범위 밖으로 남겨둔다.

## 해석

- **시차 버그 수정은 명세 변경이 아니라 정합성 수정이었고, 결과가 이를 뒷받침한다.**
  세 투자자 전원에서 persist 계수가 50~130% 커졌고, 예상(사전 정확 Lasso)과 실측이
  소수 3자리 수준으로 일치했다 — 우연한 개선이 아니라 문서화된 정의를 제대로
  구현했을 때 나오는 결과다.
- **기관의 서사가 바뀐다.** 원고 §4.4가 기관을 "식별된 선호가 약하거나 없음"으로
  다루고 있었다면, lag0에서는 OOS R²가 양수로 전환되고 persist가 V1/V2/V3 전부
  통과하는 강한 신호로 격상된다.
- **개인 α KOSPI의 부호 반전이 가장 큰 서술 변화다.** 국면 반응 방향 자체가
  바뀌므로 원고에서 이 부호로 서술한 문장은 전부 재검토가 필요하다.
- **원인 규명은 여기서 멈춘다.** 왜 시차를 하루 당기면 기관 persist의 walk-forward
  반전이 사라지는지(단순히 계수가 커져서 잡음 대비 신호비가 개선된 것인지, 아니면
  `a_t`가 진짜 정보를 담고 있어서인지)는 분해하지 않았다 — 다음 액션 참조.

## 다음 액션

1. **사용자 판단 필요 — lag0을 새 canonical로 채택할지 결정.** 채택 시 원고 §3.3
   특징 정의, §4.4 기관 서술, 개인 α KOSPI 부호, 초록의 관련 수치를 전부 갱신해야
   한다(이번 작업 범위 밖, 원고는 손대지 않았음).
2. 채택 시 `docs/persist_spec_and_results.md`에 lag0 채택 경위 추가.
3. 기관 persist의 walk-forward 반전 소멸이 계수 크기 증가(신호비 개선)만으로
   설명되는지, 아니면 정보 시점 정렬 자체의 효과인지 별도 진단 필요(예: lag0
   계수를 인위적으로 lag1 크기로 스케일 다운해 반전이 재현되는지 확인).
4. `remove_persist` ablation이 기관·개인까지 유의해진 점을 §4.2 표에 반영할지 결정.
5. 현대차 전이 실험 — 데이터·스크립트 확보 필요, 별도 작업.
