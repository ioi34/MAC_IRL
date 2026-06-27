# Parkinson 컨텍스트 E5 환율·KOSPI 통합

- 날짜/시간: 2026-06-27 17:46 KST
- 목적: Parkinson 표준 모델에서 네 컨텍스트의 결합 효과 확인
- 변경한 것: 환율 1일·5일, KOSPI200 1일·20일을 동시에 적용
- 고정 조건: E0와 동일한 보상특징, 데이터, CPCV, seed, 학습 설정
- 데이터: `data/processed/dataset_context_binary.npz`
- 설정 파일: `experiments/2026-06-27/configs/context_parkinson_e5_all.yaml`
- 결과 폴더: `runs/context_parkinson_e5_all/`
- 주요 결과: 외국인 NLL 0.695098·Macro F1 0.616784·개선 split 28.9%, 기관 0.715910·0.521082·28.9%, 개인 0.711551·0.569536·22.2%
- 해석: 세 투자자 모두 평균 NLL이 크게 악화되어 통합 모델은 채택하지 않음. 현재 표본에서 파라미터 증가에 따른 과적합 가능성이 큼
- 다음 액션: 외국인 KOSPI200 1일 단일 모델을 우선 후보로 채택
