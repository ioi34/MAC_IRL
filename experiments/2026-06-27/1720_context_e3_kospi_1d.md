# 컨텍스트 E3 KOSPI200 1일

- 날짜/시간: 2026-06-27 17:20 KST
- 목적: KOSPI200 1일 수익률이 투자자별 보상 가중치를 조정할 때 행동 재현 성능 확인
- 변경한 것: `kospi_return_1d` 하나로 `w_t = beta + B C_t` 적용
- 고정 조건: E0와 동일한 데이터, 특징, binary 라벨, CPCV 45 splits, seed 42, 200 epochs, L1 0.001
- 데이터: `data/processed/dataset_context_binary.npz`
- 설정 파일: `experiments/2026-06-27/configs/context_e3_kospi_1d.yaml`
- 결과 폴더: `runs/context_e3_kospi_1d/`
- 주요 결과: 외국인 NLL 0.650957·Macro F1 0.612443·개선 split 24.4%, 기관 0.699819·0.529100·46.7%, 개인 0.685609·0.574473·8.9%
- 해석: 기관 Macro F1은 상승했지만 세 투자자 모두 평균 NLL이 악화되고 과반 split 개선 조건도 충족하지 못함
- 다음 액션: KOSPI200 20일 누적수익률과 비교
