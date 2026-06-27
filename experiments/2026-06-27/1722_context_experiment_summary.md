# 환율·KOSPI 조건부 가중치 실험 종합

- 날짜/시간: 2026-06-27 17:22 KST
- 목적: E0~E5의 실제 CPCV 결과를 비교해 유효한 컨텍스트 판정
- 변경한 것: 고정 가중치, 단일 컨텍스트 4개, 네 컨텍스트 통합 모델 비교
- 고정 조건: 973개 표본, underwater + herd + momentum, binary 라벨, 동일 CPCV split·seed·학습 설정
- 데이터: `data/processed/dataset_context_binary.npz`
- 설정 파일: `experiments/2026-06-27/configs/features_context_no_volatility.yaml`, `experiments/2026-06-27/configs/context_e*.yaml`
- 결과 폴더: `runs/context_e0_baseline/` ~ `runs/context_e5_all/`
- 주요 결과: 기관의 `fx_return_5d`만 평균 NLL 개선과 과반 split 개선을 동시에 충족. NLL 0.697315 → 0.696820, Macro F1 0.522122 → 0.529124, 개선 split 60.0%
- 해석: 단기 하루 변화보다 5일 환율 국면이 기관 행동 가중치 변화에 제한적으로 유효했다. KOSPI 컨텍스트와 4개 통합 모델은 채택 기준 미달이며 통합 모델은 과적합 신호가 큼
- 다음 액션: 기관 FX 5일 효과의 기간별 안정성, L1 민감도, 컨텍스트 계수의 경제적 해석을 후속 검증

## 검증

- 컨텍스트·binary 라벨·모델 shape·B=0 동등성·L1·train-only scaling·스키마 테스트: 27개 통과
- `git diff --check`, Python compileall 통과

## 근거 구분

- 실제 파일 기반: 각 `runs/context_e*/cv_metrics.csv`, `cv_metrics_summary.csv`, `context_weights_summary.csv`
- 과거 결과 기반: E0 재현 비교에 `runs/exp_20260626_single_herd_no_volatility_2021_current_label/` 사용
- 추론: 통합 모델 과적합 가능성은 파라미터 증가와 전체 투자자의 OOS NLL 악화를 함께 본 해석
