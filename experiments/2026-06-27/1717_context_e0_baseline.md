# 컨텍스트 E0 고정 가중치 기준선 재현

- 날짜/시간: 2026-06-27 17:17 KST
- 목적: 환율·KOSPI 컨텍스트 실험 전에 기존 최고 binary 기준선을 동일하게 재현하는지 확인
- 변경한 것: 컨텍스트 4개를 전처리 결과에 저장했지만 모델 가중치에는 사용하지 않음
- 고정 조건: underwater + herd(window=5) + momentum(window=20), binary rolling median, 973개 표본, CPCV 45 splits, seed 42, 200 epochs, L1 0.001
- 데이터: `samsung_macirl_EXTENDED_2019_2025.csv` 중 2021-01-04~2025-12-29, 학습 표본 2022-01-06~2025-12-29
- 설정 파일: `configs/data_context.yaml`, `experiments/2026-06-27/configs/features_context_no_volatility.yaml`, `experiments/2026-06-27/configs/context_e0_baseline.yaml`
- 결과 폴더: `runs/context_e0_baseline/`
- 주요 결과: 외국인 NLL 0.647088, 기관 0.697315, 개인 0.679515. 기존 `runs/exp_20260626_single_herd_no_volatility_2021_current_label/`의 모든 Accuracy, Macro F1, NLL과 차이 0.000000
- 해석: ±0.002 재현 게이트를 통과했으며 컨텍스트 실험의 기준선으로 사용할 수 있음
- 다음 액션: E1~E5 단일·통합 컨텍스트 CPCV 실행
