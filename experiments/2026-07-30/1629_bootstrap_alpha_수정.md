# 부트스트랩이 α(컨텍스트 주효과)를 누락하던 버그 수정 및 재실행

- 날짜/시간: 2026-07-30 16:10~16:25 (재실행), 16:29 (기록)
- 목적: `scripts/run_continuous_reward_validation.py`의 월별 블록 부트스트랩이
  `bootstrap_context_weights.csv`(B, 특징×맥락 상호작용)만 저장하고 α(컨텍스트
  주효과)는 전혀 저장하지 않았다. 원고 §3.3은 α를 해석 대상 다섯 계수에 포함시키고
  기여 ③은 α로 "외국인·개인이 시장 국면 반응에서도 대립한다"를 주장하므로, 이
  주장에 부트스트랩 신뢰구간 근거가 없는 상태였다. 근본 원인을 고치고 재실행한다.
- 데이터: `data/processed/dataset_continuous_reward3_persist.npz`
- 설정 파일: `configs/data_continuous.yaml`, `configs/features_continuous_reward3_persist.yaml`,
  `configs/model.yaml`, `configs/train.yaml`, `configs/experiment_continuous_reward3_persist_epochs75.yaml`
- 결과 폴더: `runs/continuous_reward3_persist_epochs75_validation/weight_bootstrap/`
  (부트스트랩만 재실행, ablation·ridge·walk-forward는 완료 상태라 resume으로 스킵됨)

## 근본 원인

`_run_monthly_block_bootstrap`(및 `scripts/train_continuous_walk_forward.py`의
walk-forward 학습부)이 매 resample/윈도우마다 새로 만드는
`ContinuousInvestorIRLModel(...)` 생성자에 `context_main_effect` 인자를 **전달하지
않고 있었다.** 이 인자의 기본값은 `False`(`src/models/continuous.py:47`)이므로,
config가 `model.context_main_effect: true`를 지정해도 부트스트랩·walk-forward에서는
**α 파라미터 자체가 모델에 존재하지 않는 채로 학습됐다** — 단순히 "저장을 안 한 것"이
아니라 "애초에 추정하지 않은 것"이었다. 원 학습(`scripts/train_continuous.py`)만
`context_main_effect=context_main_effect`를 올바르게 전달하고 있어 이 버그를
피해갔다(그래서 `runs/*/context_main_weights.csv`는 정상 존재).

## 할 일 1 — 부트스트랩 수정 (`scripts/run_continuous_reward_validation.py`)

- `_run_monthly_block_bootstrap`에 `context_main_effect = bool(config.get("model", {}).get("context_main_effect", False))` 추가.
- 모델 생성 시 `context_main_effect=context_main_effect` 전달.
- 매 investor 적합 후 `model.context_main`이 `None`이 아니면 (resample, investor,
  context, weight) 행을 별도 리스트에 축적.
- 루프 종료 후 `bootstrap_context_main_weights.csv`(원자료)와
  `bootstrap_context_main_weights_summary.csv`(기존 `_weight_summary`를
  `["investor", "context"]`로 그룹화해 재사용 — 열 구성은 기존 요약 파일과 동일)로 저장.
  기존 `bootstrap_context_weights.csv`(B) 파서를 건드리지 않도록 **완전히 별도 파일**로 뺐다.

## 할 일 2 — walk-forward 확인 (수정 없음, 사실 확인만)

`scripts/train_continuous_walk_forward.py:168`의 `ContinuousInvestorIRLModel(...)`
호출도 동일하게 `context_main_effect`를 전달하지 않는다 — 같은 근본 원인이 walk-forward
에도 있다. `runs/continuous_reward3_persist_epochs75_validation/walk_forward/`
전체를 확인했지만 `context_main_weights.csv`가 어디에도 없다(연도별 `year_*/` 폴더
포함). 즉 walk-forward는 α를 **한 번도 추정한 적이 없다.** 지시대로 이번엔 고치지
않고 사실만 기록한다 — `analysis/walk_forward_context_stability.csv`가 B 행만
갖는 것은 결측이 아니라 애초에 α를 계산한 적이 없기 때문이다.

## 할 일 3 — 부트스트랩만 재실행

`weight_bootstrap/bootstrap_reward_weights.csv`만 삭제 후 동일 명령 재실행.
로그로 확인: ablation 8종·ridge 15종·walk-forward는 전부
`skip completed ...`로 스킵됐고 부트스트랩(200회)만 재실행됐다(약 15분,
16:10~16:25). `--force`는 사용하지 않았다.

## 검증 — bootstrap_context_main_weights_summary.csv

정확히 **6행**(3 투자자 × 2 맥락)이다.

| 대상 | 맥락 | 부트스트랩 평균 [95% CI] | 방향일관성 | CPCV 45split α (참고) | CI가 0 배제? |
| --- | --- | --- | ---: | --- | :---: |
| 외국인 | kospi_return_1d | +0.0325 [+0.0178, +0.0483] | 100% | +0.0331 (100%) | **예** |
| 외국인 | fx_level_z_252 | −0.0225 [−0.0479, −0.0002] | 98.0% | −0.0215 (97.8%) | **예**(경계 근접) |
| 기관 | kospi_return_1d | −0.0061 [−0.0119, −0.0002] | 98.0% | −0.0060 (100%) | **예**(경계 근접) |
| 기관 | fx_level_z_252 | −0.0038 [−0.0122, +0.0003] | 88.0% | −0.0035 (97.8%) | **아니오** |
| 개인 | kospi_return_1d | −0.0217 [−0.0424, −0.0059] | 100% | −0.0216 (100%) | **예** |
| 개인 | fx_level_z_252 | +0.0322 [+0.0017, +0.0643] | 99.0% | +0.0328 (100%) | **예**(경계 근접) |

출처: `runs/continuous_reward3_persist_epochs75_validation/weight_bootstrap/bootstrap_context_main_weights_summary.csv`

→ **부트스트랩 평균이 CPCV 45split α와 소수 3~4자리까지 일치** — 버그 수정이
올바르게 작동함을 확인했다. **6개 중 5개가 CI로 0을 배제**한다(단, 외국인 fx·
기관 kospi·개인 fx는 CI 경계가 0에 상당히 가깝다). **기관의 fx_level_z_252만
CI가 0을 포함해 부트스트랩 기준으로는 유의하지 않다** — CPCV 점추정 부호 일관성은
97.8%로 높았지만, 월별 블록 재표본 하에서는 방향일관성이 88%로 떨어진다(90% 문턱
미달, `passes_sign_90=False`).

**기여 ③의 핵심 주장(외국인·개인이 시장 국면 반응에서 대립)은 이제 부트스트랩
근거를 갖는다** — 외국인 kospi(+)/개인 kospi(−), 외국인 fx(−)/개인 fx(+) 네
계수 모두 CI가 0을 배제하고 부호가 정확히 반대다.

## ⚠️ 중요 — 이전 세션(1536 노트) 결론 정정

α를 포함해 모델을 올바르게 재적합하자 **β(보상가중치) 부트스트랩 결과도 함께
바뀌었다.** α가 빠진 채(버그 상태) 적합했을 때는 α가 설명해야 할 변동의 일부를
β·B가 대신 흡수했기 때문이다. 9개 β 중 8개는 유의성 판정(`ci_excludes_zero`)이
그대로였지만, **개인 underwater 하나가 뒤집혔다**:

| 대상 | 피처 | 버그 상태(α 누락) 평균 [CI] | 수정 후 평균 [CI] | 판정 변화 |
| --- | --- | --- | --- | --- |
| 개인 | underwater | +0.0413 [+0.0100, +0.0695] **배제** | +0.0291 [−0.0028, +0.0581] **미배제** | **유의 → 비유의** |

나머지 8개(외국인 momentum/persist/underwater, 기관 momentum/persist/underwater,
개인 momentum/persist)는 방향과 유의성 판정이 모두 유지됐다(수치는 소폭 이동).

**정정 대상**: `experiments/2026-07-30/1536_persist_epochs75_검증스위트.md`의
"Bootstrap 95% CI" 표에서 "개인 underwater: 구·신 canonical 모두 CI가 0을 계속
배제" 서술은 **버그가 있던 부트스트랩 결과에 근거한 것으로, 지금 수치로는 성립하지
않는다.** 개인 underwater는 `passes_sign_90`은 여전히 True(95.5%)이므로 부호
자체는 안정적이지만, 엄격한 CI 기준으로는 더 이상 "0을 배제"라고 말할 수 없다.
(참고로 `1536` 노트가 대조했던 "구 canonical"(`runs/continuous_reward3_persist_validation`)
쪽 부트스트랩은 이번 수정 대상이 아니므로 그쪽 수치는 손대지 않았다 — 거기도
같은 버그가 있었을 것이므로 재비교하려면 별도로 재실행이 필요하다.)

출처: `runs/continuous_reward3_persist_epochs75_validation/weight_bootstrap/bootstrap_reward_weights_summary.csv`

## 해석

- **버그의 성격은 "미저장"이 아니라 "미추정"이었다.** 부트스트랩 모델이
  `context_main_effect`를 안 받아 α 파라미터 자체가 없었으므로, 지금까지의
  부트스트랩은 config가 명시한 모형과 다른(더 단순한) 모형을 적합해온 것이다.
  이번 수정으로 부트스트랩이 학습·CPCV와 동일한 모형 명세를 쓰게 됐다.
- **기여 ③의 α 주장은 이제 근거가 있다.** 외국인·개인의 kospi·fx α 네 개 모두
  부트스트랩 CI가 0을 배제하고 부호가 정반대다.
- **기관의 α는 약하다.** kospi는 경계선에서 유의하지만 fx는 유의하지 않다 —
  기관을 α 관련 주장에 포함시키려면 이 비대칭을 명시해야 한다.
- **β 쪽 부수 효과(개인 underwater)를 원고에 반영해야 한다.** 이 계수를
  "부트스트랩으로 확인된 손실회피"로 인용하는 부분이 있다면 정정이 필요하다 —
  `passes_sign_90`(부호 안정성)은 유지되지만 `ci_excludes_zero`(구간 추정 유의성)는
  깨졌다.
- walk-forward는 α를 아예 추정하지 않으므로, walk-forward 기반으로 α의 안정성을
  주장하는 문장이 원고에 있다면 그 근거가 존재하지 않는다.

## 다음 액션

1. `1536_persist_epochs75_검증스위트.md`에 이 노트를 참조하는 정정 각주 추가
   (또는 해당 표 직접 수정) — 개인 underwater 판정 변경 반영.
2. 원고에서 개인 underwater를 "부트스트랩 CI로 확인" 식으로 인용한 곳이 있는지
   확인하고, 있으면 "부호 안정성(passes_sign_90)은 확인, CI 유의성은 미확인"으로 수정.
3. 기여 ③ 서술에 기관 fx_level_z_252 α의 비유의성을 명시할지 결정.
4. (선택, 미착수) `train_continuous_walk_forward.py`도 동일하게
   `context_main_effect`를 전달하도록 고치면 walk-forward에서도 α 안정성을
   볼 수 있다 — 이번 요청 범위 밖이라 보류.
5. (선택, 미착수) 구 canonical(`runs/continuous_reward3_persist_validation`)의
   부트스트랩도 같은 버그를 안고 있을 가능성이 높다 — 재비교가 필요하면 별도 재실행.
