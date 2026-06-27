# 컨텍스트 E5 환율·KOSPI 통합

- 날짜/시간: 2026-06-27 17:21 KST
- 목적: 환율 1일·5일과 KOSPI200 1일·20일 컨텍스트의 결합 효과 확인
- 변경한 것: 네 컨텍스트를 동시에 사용해 `w_t = beta + B C_t` 적용
- 고정 조건: E0와 동일한 데이터, 특징, binary 라벨, CPCV 45 splits, seed 42, 200 epochs, L1 0.001
- 데이터: `data/processed/dataset_context_binary.npz`
- 설정 파일: `experiments/2026-06-27/configs/context_e5_all.yaml`
- 결과 폴더: `runs/context_e5_all/`
- 주요 결과: 외국인 NLL 0.688840·Macro F1 0.600819·개선 split 11.1%, 기관 0.705579·0.545952·40.0%, 개인 0.703897·0.565965·20.0%
- 해석: 기관 Macro F1은 상승했지만 세 투자자 모두 평균 NLL이 크게 악화됨. 4개 동시 투입은 현재 표본에서 과적합 가능성이 높아 채택하지 않음
- 다음 액션: 기관 `fx_return_5d` 단일 모델을 핵심 후보로 두고 규제 강도 및 기간 안정성 검증
