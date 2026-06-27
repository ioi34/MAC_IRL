# Parkinson 컨텍스트 E3 KOSPI200 1일

- 날짜/시간: 2026-06-27 17:44 KST
- 목적: Parkinson 표준 모델에서 KOSPI200 1일 수익률의 조건부 가중치 효과 확인
- 변경한 것: `kospi_return_1d` 하나로 `w_t = beta + B C_t` 적용
- 고정 조건: E0와 동일한 보상특징, 데이터, CPCV, seed, 학습 설정
- 데이터: `data/processed/dataset_context_binary.npz`
- 설정 파일: `experiments/2026-06-27/configs/context_parkinson_e3_kospi_1d.yaml`
- 결과 폴더: `runs/context_parkinson_e3_kospi_1d/`
- 주요 결과: 외국인 NLL 0.654749·Macro F1 0.621962·개선 split 75.6%, 기관 0.701463·0.514403·48.9%, 개인 0.690264·0.573769·57.8%
- 해석: 외국인은 NLL, Macro F1, Accuracy가 모두 개선되어 명확히 판정 기준 통과. 개인은 과반 split과 F1은 개선됐지만 평균 NLL이 악화되어 기준 미달
- 다음 액션: 외국인 핵심 컨텍스트 후보로 유지하고 20일·통합 모델과 비교
