# Parkinson 컨텍스트 E4 KOSPI200 20일

- 날짜/시간: 2026-06-27 17:45 KST
- 목적: Parkinson 표준 모델에서 KOSPI200 20일 누적 로그수익률의 조건부 가중치 효과 확인
- 변경한 것: `kospi_return_20d` 하나로 `w_t = beta + B C_t` 적용
- 고정 조건: E0와 동일한 보상특징, 데이터, CPCV, seed, 학습 설정
- 데이터: `data/processed/dataset_context_binary.npz`
- 설정 파일: `experiments/2026-06-27/configs/context_parkinson_e4_kospi_20d.yaml`
- 결과 폴더: `runs/context_parkinson_e4_kospi_20d/`
- 주요 결과: 외국인 NLL 0.687511·Macro F1 0.602794·개선 split 42.2%, 기관 0.705743·0.493628·28.9%, 개인 0.694883·0.570915·44.4%
- 해석: 세 투자자 모두 평균 NLL이 악화되어 컨텍스트 효과 없음
- 다음 액션: 네 컨텍스트 통합 효과 확인
