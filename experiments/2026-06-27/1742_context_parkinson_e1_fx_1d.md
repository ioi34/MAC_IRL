# Parkinson 컨텍스트 E1 환율 1일

- 날짜/시간: 2026-06-27 17:42 KST
- 목적: Parkinson 표준 모델에서 USD/KRW 1일 로그수익률의 조건부 가중치 효과 확인
- 변경한 것: `fx_return_1d` 하나로 `w_t = beta + B C_t` 적용
- 고정 조건: E0와 동일한 보상특징, 데이터, CPCV, seed, 학습 설정
- 데이터: `data/processed/dataset_context_binary.npz`
- 설정 파일: `experiments/2026-06-27/configs/context_parkinson_e1_fx_1d.yaml`
- 결과 폴더: `runs/context_parkinson_e1_fx_1d/`
- 주요 결과: 외국인 NLL 0.656072·Macro F1 0.602981·개선 split 51.1%, 기관 0.702322·0.504231·42.2%, 개인 0.686473·0.565359·17.8%
- 해석: 외국인은 NLL과 과반 split 조건을 간신히 충족했지만 Macro F1은 하락해 약한 효과로 판단. 기관과 개인은 기준 미달
- 다음 액션: 다른 단일 컨텍스트와 개선 폭 비교
