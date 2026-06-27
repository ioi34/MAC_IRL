# Parkinson 컨텍스트 E2 환율 5일

- 날짜/시간: 2026-06-27 17:43 KST
- 목적: Parkinson 표준 모델에서 USD/KRW 5일 로그수익률의 조건부 가중치 효과 확인
- 변경한 것: `fx_return_5d` 하나로 `w_t = beta + B C_t` 적용
- 고정 조건: E0와 동일한 보상특징, 데이터, CPCV, seed, 학습 설정
- 데이터: `data/processed/dataset_context_binary.npz`
- 설정 파일: `experiments/2026-06-27/configs/context_parkinson_e2_fx_5d.yaml`
- 결과 폴더: `runs/context_parkinson_e2_fx_5d/`
- 주요 결과: 외국인 NLL 0.663677·Macro F1 0.597288·개선 split 31.1%, 기관 0.700471·0.511065·53.3%, 개인 0.688927·0.552562·26.7%
- 해석: 기관은 과반 split에서 개선됐지만 평균 NLL이 기준선보다 높아 판정 기준 미달. 외국인과 개인도 효과 없음
- 다음 액션: KOSPI 단일 컨텍스트와 비교
