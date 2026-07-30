# persist epochs75(신 canonical) 검증 스위트 실행 — 구 canonical과 대조

- 날짜/시간: 2026-07-30 15:20~15:32 (실행), 15:36 (기록)
- 목적: `experiments/2026-07-30/1450_HP통일_수렴검정.md`에서 기존 canonical
  (`runs/continuous_reward3_persist`)이 기관·개인에서 수렴하지 않았고, epoch만 75로
  통일한 `runs/continuous_reward3_persist_epochs75`가 새 canonical로 채택됐다.
  학습은 끝났고 검증 스위트(bootstrap 200 / walk-forward / ridge / ablation)만 남아
  있었다. 이를 정식 실행하고 구 canonical의 검증 결과와 대조한다.
- 변경한 것: **없음.** 이미 학습된 epoch75 모델에 대해 검증 스위트만 신규 실행.
- 고정 조건: 3특징(momentum·persist·underwater), 맥락 2개, κ=1, λ=0.005 통일,
  CPCV 10/2/purge1/embargo5 = 45 split, seed 42, n=973. (구 canonical과 동일 조건,
  차이는 기관·개인 epoch만 75로 통일된 것.)
- 데이터: `data/processed/dataset_continuous_reward3_persist.npz`
- 설정 파일: `configs/data_continuous.yaml`, `configs/features_continuous_reward3_persist.yaml`,
  `configs/model.yaml`, `configs/train.yaml`, `configs/experiment_continuous_reward3_persist_epochs75.yaml`
- 결과 폴더:
  - 신 canonical 검증: `runs/continuous_reward3_persist_epochs75_validation/`
  - 구 canonical 검증(대조군): `runs/continuous_reward3_persist_validation/`
  - 사후 검증 수치: `experiments/2026-07-30/verify_paper_numbers_epochs75/`

## 실행 메모

- 시작 전 `runs/continuous_reward3_persist_epochs75_validation/ablation/remove_momentum`을
  삭제했다 — 이 디렉터리는 split_00~06이 15:13, split_07~44가 15:17로 타임스탬프가
  갈라져 있어 서로 다른 두 번의 실행이 섞인 것으로 판단(중단 후 재개 이력). 반면
  `remove_persist`는 15/45에서 끊긴 단일 미완료 실행이라 그대로 두고 resume에 맡겼다.
- `scripts/run_continuous_reward_validation.py`를 `--bootstrap-resamples 200`으로
  1회 호출, 끊지 않고 완주(15:20~15:32, CPCV 재실행분 포함 약 12분). ablation 8종
  (baseline + 7 제거) · ridge 15단계 · walk-forward 3개년 · bootstrap 200회 모두 정상 종료.
- `--config-dir experiments/2026-07-30/configs/persist_epochs75_validation`을 명시했다 —
  이 스크립트의 `--config-dir` 기본값이 과거 다른 실험(2026-07-28)의 커밋된 config
  폴더를 가리키는 문제가 있었던 걸 이전 세션에서 발견했기 때문에, 이번엔 전용 경로를
  줘서 충돌을 피했다. `git status`로 재확인, 무관한 파일 변경 없음.
- `scripts/analyze_continuous_reward_validation.py`는 마지막 단계(`momentum`↔`relative`
  상관 분리 분석)에서 `FileNotFoundError`로 실패했다. 이 스크립트는 5특징(`relative`
  포함) 구성 전용으로 하드코딩돼 있어 3특징 persist 구성에는 애초에 적용 불가 —
  구 canonical 검증 때도 동일하게 실패했던 **기존 한계**이며 이번 실행과 무관하다.
  그 앞에 저장되는 13개 분석 파일(VIF·상관·ablation·ridge·walk-forward)은 정상 생성됐다.

## 검증 스위트 대조 — 구 canonical vs 신 canonical(epoch75)

### 1) Bootstrap 95% CI (월별 블록, 200회) — 0을 배제하는 계수

| 대상 | 피처 | 구 canonical CI | 구 결과 | 신(epoch75) CI | 신 결과 |
| --- | --- | --- | :---: | --- | :---: |
| 외국인 | momentum | [0.0306, 0.0750] | 배제 | [0.0306, 0.0750] | 배제(동일) |
| 외국인 | persist | [0.0310, 0.0718] | 배제 | [0.0310, 0.0718] | 배제(동일) |
| 외국인 | underwater | [-0.0185, 0.0213] | 미배제 | [-0.0185, 0.0213] | 미배제(동일) |
| 기관 | momentum | [-0.0048, 0.0041] | 미배제 | [-0.0096, 0.0023] | 미배제 |
| 기관 | persist | [0.0002, 0.0049] | **배제** | [0.0001, 0.0208] | **배제** |
| 기관 | underwater | [-0.0048, 0.0021] | 미배제 | [-0.0105, 0.0029] | 미배제 |
| 개인 | momentum | [-0.0347, -0.0279] | 배제 | [-0.0643, -0.0174] | 배제 |
| 개인 | persist | [0.0279, 0.0357] | 배제 | [0.0243, 0.0722] | 배제 |
| 개인 | **underwater** | [0.0258, 0.0353] | **배제** | [0.0100, 0.0695] | **배제** |

출처: `runs/continuous_reward3_persist_validation/weight_bootstrap/bootstrap_reward_weights_summary.csv`,
`runs/continuous_reward3_persist_epochs75_validation/weight_bootstrap/bootstrap_reward_weights_summary.csv`

→ **외국인은 소수점까지 완전 동일**(canonical에서 이미 75 epoch였으므로 통제군 역할).
**0-배제 여부는 9개 계수 전부 구·신 동일** — 어느 계수도 유의성이 뒤집히지 않았다.
개인 underwater는 구·신 모두 CI가 0을 계속 배제한다.

### 2) Walk-forward (2023/2024/2025 연도별 재학습) 부호 반전

| 대상 | 피처 | 구 반전 | 신 반전 | 비고 |
| --- | --- | :---: | :---: | --- |
| 외국인 | 전 피처 | momentum/persist 없음, underwater 있음 | 동일 | 통제군 |
| 기관 | momentum | 있음 | 있음 | 변화 없음 |
| 기관 | persist | **있음**(1/3 창 음전환) | **있음**(1/3 창 음전환) | 계수 작아 불안정 지속 |
| 기관 | underwater | 있음 | **없음**(개선) | 3/3 양(+)으로 전환 |
| 개인 | momentum/persist | 없음 | 없음 | 유지 |
| 개인 | underwater | 있음 | 있음 | 변화 없음 |

출처: `runs/continuous_reward3_persist_validation/analysis/walk_forward_reward_stability.csv`,
`runs/continuous_reward3_persist_epochs75_validation/analysis/walk_forward_reward_stability.csv`

→ 기관 persist는 여전히 연도별 재학습에서 1/3 구간 부호가 흔들린다(계수가
다른 피처 대비 작아 잡음에 민감) — bootstrap CI는 0을 배제하지만 walk-forward는
약함, 두 검증의 성격이 다르므로(블록 재표본 vs 완전 재학습) 모순은 아니다.

### 3) Ridge(선택 λ) vs Lasso 부호 일치

구·신 모두 **9/9 전 계수 부호 일치**(`sign_agreement=True`), 변화 없음.
다만 선택된 ridge 강도(λ)는 기관 10.0→3.0, 개인 0.3→1.0으로 바뀌었다(계수 크기가
커진 만큼 최적 정규화 강도도 이동) — 외국인은 1.0으로 동일.
출처: `.../analysis/ridge_lasso_reward_comparison.csv`, `.../analysis/ridge_selected.csv`

### 4) Ablation paired bootstrap — 유의한 저하/개선

| variant | investor | metric | 구 canonical | 신(epoch75) |
| --- | --- | --- | --- | --- |
| remove_persist | 외국인 | rmse | **유의 저하** | **유의 저하**(유지) |
| remove_persist | 외국인 | correlation | **유의 저하** | **유의 저하**(유지) |
| remove_momentum | 기관 | rmse | 유의 개선 | 유의 개선(유지) |
| remove_momentum | 기관 | correlation | 유의 개선 | 유의 개선(유지) |
| remove_traditional_group | 기관 | rmse/correlation | 유의 개선 | 유의 개선(유지) |
| remove_behavioral_group | 개인 | direction_accuracy | 유의 개선 | — (사라짐) |
| remove_behavioral_group | 개인 | correlation | — | **유의 저하**(신규) |
| remove_underwater | 개인 | direction_accuracy | 유의 개선 | — (사라짐) |
| remove_underwater | 개인 | correlation | — | **유의 저하**(신규) |

출처: `runs/continuous_reward3_persist_validation/analysis/ablation_paired_bootstrap.csv`,
`runs/continuous_reward3_persist_epochs75_validation/analysis/ablation_paired_bootstrap.csv`

→ **remove_persist 외국인은 구·신 통틀어 유일한 "저하" 유의 항목** — 질문했던 대로
변하지 않았다. **기관 momentum 제거는 여전히 유의한 개선**(유지) — 질문했던 대로다.
가장 눈에 띄는 변화는 개인 쪽: 구 canonical에서는 behavioral_group·underwater를
제거하면 개인 **방향정확도**가 역설적으로 유의하게 좋아졌는데, epoch75 수렴 후에는
이 효과가 사라지고 대신 **상관이 유의하게 나빠진다**(제거하면 손해 — 방향이 상식과
일치). 미수렴 모델의 반직관적 결과가 수렴 후 정상적인 방향으로 정리된 것으로 보인다.

## 사후 검증 수치 (`scripts/verify_paper_numbers.py`, epoch75)

| 항목 | 외국인 | 기관 | 개인 |
| --- | ---: | ---: | ---: |
| 설계행렬 rank (열11 만계수) | 135/135 결손 0 | 〃 | 〃 |
| cond(W) 최대 | 6.94 (split 39, retail에서 최악) | | |
| OOS R²(기준=학습평균) | +0.1301 | −0.0033 | +0.1068 |
| 행동 AR(1) | +0.4012 | +0.1489 | +0.3445 |
| 수렴: 앞80% 감소분(최소) | 0.9174 | **0.9908** | **0.9960** |
| 수렴: 마지막step 비중(최대) | 0.0157 | **0.0043** | **0.0014** |
| 수렴: θ 마지막변화 상대(최대) | 3.93% | 1.70% | 1.21% |

출처: `experiments/2026-07-30/verify_paper_numbers_epochs75/{design_matrix_uniqueness,out_of_sample_r2_summary,action_autocorrelation,convergence_summary}.csv`.
`1450_HP통일_수렴검정.md`가 보고한 수치와 완전히 일치 — 재현 확인.

## 가중치 결과 (baseline, 45 split 평균 ± 표준편차, 부호 일관성)

| 대상 | 피처 | 값 | 일관성 | 출처 |
| --- | --- | ---: | ---: | --- |
| 외국인 | momentum | +0.04354 ± 0.00903 | 100% | `runs/continuous_reward3_persist_epochs75_validation/ablation/baseline/reward_weights_summary.csv` |
| 외국인 | persist | +0.05055 ± 0.00604 | 100% | 〃 |
| 외국인 | underwater | +0.00612 ± 0.00476 | 91.1% | 〃 |
| 기관 | momentum | −0.00148 ± 0.00223 | 75.6%(음) | 〃 |
| 기관 | persist | +0.01024 ± 0.00307 | 100% | 〃 |
| 기관 | underwater | −0.00028 ± 0.00055 | 75.6%(음) | 〃 |
| 개인 | momentum | −0.03556 ± 0.00662 | 100% | 〃 |
| 개인 | persist | +0.04620 ± 0.00778 | 100% | 〃 |
| 개인 | underwater | +0.02883 ± 0.01247 | 97.8% | 〃 |

(같은 폴더의 `analysis/` 산출과 `runs/continuous_reward3_persist_epochs75/reward_weights_summary.csv`
— 검증용 재학습과 원 학습 결과가 diff 없이 완전히 일치함을 확인.)

## 해석

- **검증 스위트를 통과했다.** bootstrap CI 0-배제 여부, ridge/lasso 부호, remove_persist
  외국인 유의 저하, remove_momentum 기관 유의 개선 — 질문했던 네 가지 모두 구
  canonical과 동일하게 유지된다. epoch75 교체가 헤드라인을 흔들지 않았다.
- **기관 persist는 "약하지만 실재"**: bootstrap 3/3 배제, ridge/lasso 부호 일치지만
  walk-forward 1/3 구간 부호 반전 — 계수 크기(0.01 수준)가 다른 계수의 1/5~1/10이라
  연도별 재추정에는 취약하다. 논문에서 "약한 신호"로 명시하는 게 정확하다.
- **개인의 반직관적 ablation 결과가 사라졌다.** 구 canonical에서 "behavioral 특징
  제거가 개인 방향정확도를 높인다"는 결과는 조기 종료로 미수렴한 모델의 인공물이었을
  가능성이 있다 — 수렴 후에는 제거 시 상관이 유의하게 나빠지는, 상식에 맞는 방향으로
  바뀌었다. §3.6 이후 ablation 논의가 있다면 이 변화를 반영해야 한다.
- `analyze_continuous_reward_validation.py`의 `momentum↔relative` 상관 분리 분석은
  이번에도(그리고 앞으로도) 3특징 구성에서는 실행되지 않는다 — 스크립트 수정 없이는
  해소되지 않는 기존 한계이며 본 실험의 결론에는 영향 없다.

## 다음 액션

1. epoch75를 canonical로 확정하는 문서 반영 — §3.5(수렴), §3.6(ablation, 개인
   반직관 결과 삭제/수정), 초록·기여 ③ `beyond flow persistence` 복원
   (`1450_HP통일_수렴검정.md` 다음 액션 2·3과 동일, 아직 미착수)
2. `analyze_continuous_reward_validation.py`의 `_analyze_correlated_pair`를
   피처 집합에 맞춰 조건부로 건너뛰게 고치거나 별도 옵션화 (선택 사항, 결론에는
   영향 없음)
3. `scripts/tune_hyperparameters.py` 산출물 확인해 유형별 batch·lr 튜닝 경위 기록
   (`1450_...` 다음 액션 4, 미착수)
