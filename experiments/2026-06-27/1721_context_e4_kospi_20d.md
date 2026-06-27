# 컨텍스트 E4 KOSPI200 20일

- 날짜/시간: 2026-06-27 17:21 KST
- 목적: KOSPI200 20일 누적 로그수익률이 투자자별 보상 가중치를 조정할 때 행동 재현 성능 확인
- 변경한 것: `kospi_return_20d` 하나로 `w_t = beta + B C_t` 적용
- 고정 조건: E0와 동일한 데이터, 특징, binary 라벨, CPCV 45 splits, seed 42, 200 epochs, L1 0.001
- 데이터: `data/processed/dataset_context_binary.npz`
- 설정 파일: `experiments/2026-06-27/configs/context_e4_kospi_20d.yaml`
- 결과 폴더: `runs/context_e4_kospi_20d/`
- 주요 결과: 외국인 NLL 0.674468·Macro F1 0.613152·개선 split 28.9%, 기관 0.701867·0.516501·40.0%, 개인 0.691744·0.570166·31.1%
- 해석: 세 투자자 모두 평균 NLL과 Macro F1이 기준선보다 악화되어 컨텍스트 효과 없음
- 다음 액션: 네 컨텍스트 통합 모델에서 결합 효과 확인
