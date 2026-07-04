# 투자자별 독립 하이퍼파라미터 튜닝

- 날짜/시간: 2026-07-01 17:28 KST
- 목적: 외국인·기관·개인의 손실과 optimizer를 합치지 않고 CPCV Accuracy를 최대화
- 변경한 것: `learning_rate`, `lambda_l1`, `epochs`, `batch_size`를 단계적으로 탐색하고 투자자별 override를 적용
- 고정 조건: E2 `kospi_return_1d` 컨텍스트, underwater·herd·momentum·volatility·persist, binary 행동, 973표본, CPCV 10-fold/2-test-fold(45 split), purge 1, embargo 5, `tau=1`, `weight_decay=0`
- 데이터: `data/processed/dataset_vkospi.npz`
- 설정 파일: `experiments/2026-07-01/hyperparameter_tuning/configs/final_investor_specific.yaml`
- 결과 폴더: `runs/hyperparameter_tuned_investor_specific/`
- 주요 결과: 172개 설정을 탐색했다. seed 42 기준 Accuracy는 외국인 0.6379, 기관 0.5457, 개인 0.6008이며 기존 E2 대비 세 투자자 모두 개선됐다.

## 최종 하이퍼파라미터

| 투자자 | Epoch | Batch | Learning rate | L1 |
| --- | ---: | ---: | ---: | ---: |
| 외국인 | 100 | 256 | 0.0010 | 0 |
| 기관 | 15 | 2048 | 0.0005 | 0.0100 |
| 개인 | 15 | 512 | 0.0015 | 0.0003 |

세 설정은 `investor_overrides`로 적용되며 모델·optimizer·loss는 계속 투자자별로 독립 생성된다. 세 투자자 loss를 합산하는 연산은 없다.

## 성능 결과

| 투자자 | 기존 Accuracy | 튜닝 Accuracy | 변화 | Macro F1 | NLL | Accuracy 개선 split | NLL 개선 split |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 외국인 | 0.6315 | 0.6379 | +0.0064 | 0.6220 | 0.6498 | 26/45 (57.8%) | 27/45 (60.0%) |
| 기관 | 0.5281 | 0.5457 | +0.0175 | 0.5356 | 0.6923 | 34/45 (75.6%) | 31/45 (68.9%) |
| 개인 | 0.5887 | 0.6008 | +0.0121 | 0.5881 | 0.6784 | 30/45 (66.7%) | 24/45 (53.3%) |

- 기존 출처: `runs/report_observability_vkospi_e2/cv_metrics.csv`
- 튜닝 출처: `runs/hyperparameter_tuned_investor_specific/cv_metrics.csv`
- 비교표: `experiments/2026-07-01/hyperparameter_tuning/final_comparison.csv`
- seed 42·43·44 Accuracy 평균: 외국인 0.6389, 기관 0.5457, 개인 0.6000
- seed 재검증 결과: `runs/hyperparameter_tuned_investor_specific_seed43/`, `runs/hyperparameter_tuned_investor_specific_seed44/`

## 가중치 결과

| 투자자 | 피처/가중치 | 값 | 변동성/일관성 | 출처 |
| --- | --- | ---: | ---: | --- |
| 외국인 | underwater beta | -0.0680 | std 0.0285 / 100.0% | `runs/hyperparameter_tuned_investor_specific/reward_weights_summary.csv` |
| 외국인 | herd beta | -0.1496 | std 0.0138 / 100.0% | 위와 같음 |
| 외국인 | momentum beta | 0.1351 | std 0.0139 / 100.0% | 위와 같음 |
| 외국인 | volatility beta | 0.0318 | std 0.0451 / 73.3% | 위와 같음 |
| 외국인 | persist beta | 0.0812 | std 0.0194 / 100.0% | 위와 같음 |
| 기관 | underwater beta | -0.0034 | std 0.0031 / 75.6% | 위와 같음 |
| 기관 | herd beta | -0.0059 | std 0.0023 / 95.6% | 위와 같음 |
| 기관 | momentum beta | -0.0073 | std 0.0002 / 100.0% | 위와 같음 |
| 기관 | volatility beta | 0.0066 | std 0.0019 / 95.6% | 위와 같음 |
| 기관 | persist beta | 0.0067 | std 0.0017 / 100.0% | 위와 같음 |
| 개인 | underwater beta | 0.0272 | std 0.0140 / 91.1% | 위와 같음 |
| 개인 | herd beta | -0.0411 | std 0.0008 / 100.0% | 위와 같음 |
| 개인 | momentum beta | -0.0402 | std 0.0015 / 100.0% | 위와 같음 |
| 개인 | volatility beta | -0.0214 | std 0.0165 / 86.7% | 위와 같음 |
| 개인 | persist beta | 0.0407 | std 0.0010 / 100.0% | 위와 같음 |

- 컨텍스트 가중치 출처: `runs/hyperparameter_tuned_investor_specific/context_weights_summary.csv`
- 해석: 기관은 강한 L1과 짧은 학습으로 beta 절대값이 작아졌다. Accuracy는 개선됐지만 기존 200 epoch 가중치와 직접적인 크기 비교에는 주의가 필요하다.
- 다음 액션: 최종 미사용 기간이 있다면 해당 기간을 한 번만 평가해 튜닝 선택 편향을 확인한다.

## 탐색 산출물

- 전체 1차 결과: `experiments/2026-07-01/hyperparameter_tuning/tuning_summary.csv`
- epoch/lr 탐색: `experiments/2026-07-01/hyperparameter_tuning/schedule/`
- L1 탐색: `experiments/2026-07-01/hyperparameter_tuning/long_l1/`, `experiments/2026-07-01/hyperparameter_tuning/short_l1/`
- batch 탐색: `experiments/2026-07-01/hyperparameter_tuning/long_batch/`, `experiments/2026-07-01/hyperparameter_tuning/short_batch/`
- 정밀 탐색: `experiments/2026-07-01/hyperparameter_tuning/institution_precision/`, `experiments/2026-07-01/hyperparameter_tuning/retail_precision/`, `experiments/2026-07-01/hyperparameter_tuning/institution_micro/`, `experiments/2026-07-01/hyperparameter_tuning/retail_micro/`
- 실행 로그: 각 `runs/hyperparameter_tuning/**/run.log` 및 최종 결과 폴더의 `run.log`

## 검증

- 투자자별 override·loss·학습 관찰성 테스트: 3개 통과
- 전체 테스트: 31개 통과, 기존 `herd_pairwise` registry 불일치 테스트 1개 실패
- `git diff --check`: 통과
- 통합 실행의 투자자별 `cv_metrics.csv`는 각 최고 개별 실행과 정확히 일치

## 근거 구분

- 결론 근거: `runs/`와 `experiments/2026-07-01/hyperparameter_tuning/`의 실제 실행 파일
- Notion/GitHub 메모만으로 판단한 항목: 없음
- 코드·설정 변경만으로 추론한 항목: 없음
