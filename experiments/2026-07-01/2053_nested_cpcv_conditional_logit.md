# 조건부 로짓 Nested CPCV 규제 튜닝

- 상태: **채택하지 않음 / 관련 구현·테스트·설정은 사용자 요청으로 롤백**
- 날짜/시간: 2026-07-01 20:53 KST
- 목적: optimizer 설정이 아니라 모델 복잡도인 L1 규제 강도 `C`를 투자자별로 독립 선택하고, 튜닝에 사용하지 않은 outer CPCV로 일반화 성능을 추정
- 변경한 것: 기존 reward logit과 정확히 같은 조건부 로짓 설계행렬로 변환하고 `C` 13개를 탐색. 선택 기준은 Macro F1 최고점에서 0.005 이내인 후보 중 NLL 최소
- 고정 조건: E2 `kospi_return_1d` 컨텍스트, underwater·herd·momentum·volatility·persist, 973표본, outer CPCV 10-fold/2-test-fold(45 split), inner CPCV 5-fold/2-test-fold, purge 1, embargo 5, `tau=1`
- 데이터: `data/processed/dataset_vkospi.npz`
- 설정 파일: `experiments/2026-07-01/configs/nested_cpcv_conditional_logit_f1.yaml` (롤백으로 제거됨)
- 결과 폴더: `runs/nested_cpcv_conditional_logit_f1/`
- 주요 결과: 각 outer split의 학습 구간 안에서만 투자자별 `C`를 선택했다. 외국인은 기존 E2보다 Accuracy와 Macro F1이 높았고, 기관과 개인은 유사하거나 소폭 낮았다. 세 투자자의 목적함수와 모델은 합치지 않았다.

## 선택된 하이퍼파라미터

`C`는 L1 규제 강도의 역수이므로 작을수록 강한 규제다. 아래 값은 45개 outer split에서 선택된 횟수이며, 단일값으로 요약한 최빈값은 외국인 0.03, 기관 0.1, 개인 0.03이다.

| 투자자 | 선택 C | 선택 횟수 |
| --- | ---: | ---: |
| 외국인 | 0.01 / 0.03 / 0.1 / 0.3 / 1.0 | 5 / 34 / 2 / 3 / 1 |
| 기관 | 0.03 / 0.1 / 0.3 / 1.0 | 1 / 25 / 17 / 2 |
| 개인 | 0.03 / 0.1 / 0.3 | 24 / 11 / 10 |

- 출처: `runs/nested_cpcv_conditional_logit_f1/selected_hyperparameters_summary.csv`
- 고정 solver 설정: `liblinear`, `max_iter=5000`, `tol=1e-8`, intercept 없음
- optimizer의 epoch·batch size·learning rate는 이 정확해법에서 제거했다. 이들은 수렴 방법이지 최종 모델 복잡도 하이퍼파라미터가 아니기 때문이다.

## 성능 결과

| 투자자 | 방법 | Accuracy | Macro F1 | NLL |
| --- | --- | ---: | ---: | ---: |
| 외국인 | 기존 E2 | 0.6315 | 0.6145 | 0.6526 |
| 외국인 | 기존 수동 튜닝 | 0.6379 | 0.6220 | 0.6498 |
| 외국인 | Nested CPCV | 0.6364 | 0.6181 | 0.6528 |
| 기관 | 기존 E2 | 0.5281 | 0.5167 | 0.7051 |
| 기관 | 기존 수동 튜닝 | 0.5457 | 0.5356 | 0.6923 |
| 기관 | Nested CPCV | 0.5263 | 0.5148 | 0.7007 |
| 개인 | 기존 E2 | 0.5887 | 0.5736 | 0.6894 |
| 개인 | 기존 수동 튜닝 | 0.6008 | 0.5881 | 0.6784 |
| 개인 | Nested CPCV | 0.5868 | 0.5712 | 0.6836 |

- Nested 결과: `runs/nested_cpcv_conditional_logit_f1/cv_metrics_summary.csv`
- 기존 E2: `runs/report_observability_vkospi_e2/cv_metrics_summary.csv`
- 기존 수동 튜닝: `experiments/2026-07-01/hyperparameter_tuning/final_comparison.csv`
- 기존 수동 튜닝은 동일한 outer CPCV 결과를 선택에도 사용했으므로, Nested CPCV 수치와 동일한 조건의 일반화 추정치로 간주하면 안 된다.

## NLL 단독 선택 실패 확인

NLL만 최소화한 선행 실행에서는 기관의 45개 outer split 중 39개가 `C=0.001`을 선택했다. 거의 모든 계수를 0으로 만든 모델이 NLL에서는 유리했지만 기관 Macro F1이 0.3464로 붕괴했다. 선택 기준을 바꾼 뒤 기관 Macro F1은 0.5148이 됐고 NLL은 0.6948에서 0.7007로 증가했다.

- NLL 단독 결과: `runs/nested_cpcv_conditional_logit/`
- NLL 단독 선택표: `runs/nested_cpcv_conditional_logit/selected_hyperparameters_summary.csv`
- 해석: NLL과 Macro F1 사이에 실제 trade-off가 있으므로 두 값을 임의 가중합하지 않고, Macro F1 허용구간을 먼저 정한 뒤 NLL을 보조 기준으로 사용했다.

## 가중치 결과

| 대상 | 피처/가중치 | 값 | 변동성/일관성 | 출처 |
| --- | --- | ---: | ---: | --- |
| 외국인 | underwater beta | -0.034886 | std 0.038690 / 방향 73.3% | `runs/nested_cpcv_conditional_logit_f1/reward_weights_summary.csv` |
| 외국인 | herd beta | -0.186588 | std 0.039371 / 방향 100.0% | 위와 같음 |
| 외국인 | momentum beta | 0.101679 | std 0.034118 / 방향 100.0% | 위와 같음 |
| 외국인 | volatility beta | 0.013019 | std 0.034747 / 방향 22.2% | 위와 같음 |
| 외국인 | persist beta | 0.011019 | std 0.019404 / 방향 40.0% | 위와 같음 |
| 기관 | underwater beta | 0.000536 | std 0.004888 / 방향 11.1% | 위와 같음 |
| 기관 | herd beta | -0.020279 | std 0.019822 / 방향 73.3% | 위와 같음 |
| 기관 | momentum beta | -0.052601 | std 0.022715 / 방향 100.0% | 위와 같음 |
| 기관 | volatility beta | 0.038764 | std 0.021490 / 방향 93.3% | 위와 같음 |
| 기관 | persist beta | 0.030495 | std 0.025102 / 방향 93.3% | 위와 같음 |
| 개인 | underwater beta | -0.020048 | std 0.026352 / 방향 46.7% | 위와 같음 |
| 개인 | herd beta | -0.112513 | std 0.031124 / 방향 100.0% | 위와 같음 |
| 개인 | momentum beta | -0.071136 | std 0.031243 / 방향 97.8% | 위와 같음 |
| 개인 | volatility beta | -0.015328 | std 0.023273 / 방향 51.1% | 위와 같음 |
| 개인 | persist beta | 0.067231 | std 0.032559 / 방향 100.0% | 위와 같음 |
| 외국인 | volatility × KOSPI B | 0.089186 | std 0.046866 / 방향 93.3% | `runs/nested_cpcv_conditional_logit_f1/context_weights_summary.csv` |
| 기관 | underwater × KOSPI B | 0.040711 | std 0.018371 / 방향 97.8% | 위와 같음 |
| 기관 | herd × KOSPI B | -0.032312 | std 0.018770 / 방향 97.8% | 위와 같음 |
| 기관 | volatility × KOSPI B | -0.058344 | std 0.016694 / 방향 100.0% | 위와 같음 |
| 개인 | volatility × KOSPI B | -0.069401 | std 0.057883 / 방향 93.3% | 위와 같음 |

컨텍스트 가중치 전체 실제 값은 `runs/nested_cpcv_conditional_logit_f1/context_weights_summary.csv`에 있다. 표에는 방향 일관성이 높은 항목을 기록했다.

- 해석: 외국인의 herd 음(-)·momentum 양(+), 기관의 momentum 음(-), volatility·persist 양(+), 개인의 herd·momentum 음(-)·persist 양(+)은 outer split 전체에서 방향이 안정적이었다.
- 다음 액션: E2 컨텍스트 자체도 과거 CPCV 관찰을 통해 선택됐으므로, 아키텍처까지 포함한 최종 주장에는 완전히 미사용한 시간 구간 평가가 필요하다. 현재의 1차원 `C` 탐색에는 Optuna보다 이 정확 grid가 단순하고 재현 가능하다.

## 검증

- 조건부 로짓 동치·선택 규칙·CPCV·누수·손실 테스트: 8개 통과
- 전체 테스트: 34개 통과, 기존 `herd_pairwise` registry 불일치 테스트 1개 실패
- `git diff --check`: 통과

## 근거 구분

- 결론 근거: `runs/nested_cpcv_conditional_logit*/`와 기존 비교 폴더의 실제 실행 파일
- Notion/GitHub 메모만으로 판단한 항목: 없음
- 코드·설정 변경만으로 추론한 항목: E2 아키텍처 선택까지 완전히 unbiased하지 않다는 방법론적 한계
