# Parkinson 표준 환율·KOSPI 컨텍스트 실험 종합

- 날짜/시간: 2026-06-27 17:47 KST
- 목적: Parkinson 변동성을 포함한 표준 모델에서 E0~E5의 실제 CPCV 결과 비교
- 변경한 것: 고정 가중치, 단일 컨텍스트 4개, 네 컨텍스트 통합 모델 비교
- 고정 조건: underwater + herd + momentum + Parkinson volatility, 973개 표본, binary 라벨, 동일 CPCV split·seed
- 데이터: `data/processed/dataset_context_binary.npz`
- 설정 파일: `configs/features_context.yaml`, `experiments/2026-06-27/configs/context_parkinson_e*.yaml`
- 결과 폴더: `runs/context_parkinson_e0_baseline/` ~ `runs/context_parkinson_e5_all/`
- 주요 결과: 외국인 `kospi_return_1d`가 가장 명확하게 개선. NLL 0.656193 → 0.654749, Macro F1 0.610206 → 0.621962, Accuracy 0.627925 → 0.639101, 개선 split 75.6%
- 해석: Parkinson 특징을 포함하면 외국인 행동에서 당일 시장 수익률에 따른 보상 가중치 변화가 유효했다. 환율 1일은 NLL 개선 폭이 0.000121로 작고 F1이 하락해 보조 후보에 그침. 환율 5일, KOSPI 20일, 네 변수 통합은 채택 기준 미달
- 다음 액션: 외국인 KOSPI200 1일 모델의 기간별 안정성과 규제 강도 민감도 검증

## 외국인 KOSPI200 1일 컨텍스트 계수

- underwater: -0.060312, 부호 일관성 100%
- herd: -0.068700, 부호 일관성 97.8%
- momentum: -0.103735, 부호 일관성 100%
- volatility: +0.180317, 부호 일관성 100%

## 근거 구분

- 실제 파일 기반: 각 `runs/context_parkinson_e*/cv_metrics.csv`, `cv_metrics_summary.csv`, `context_weights_summary.csv`
- 추론: 통합 모델 과적합 가능성은 파라미터 증가와 전체 투자자의 OOS NLL 악화를 함께 본 해석
- 통합 가중치 상세 해석: `experiments/2026-06-27/1746_context_parkinson_e5_all.md`

## 검증

- Parkinson high/low, 컨텍스트 누수, binary 라벨, 모델 shape, B=0 동등성, L1, train-only scaling 관련 테스트 29개 통과
- `git diff --check`, Python compileall 통과
