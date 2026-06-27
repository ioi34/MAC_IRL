# 컨텍스트 E1 환율 1일

- 날짜/시간: 2026-06-27 17:18 KST
- 목적: USD/KRW 1일 로그수익률이 투자자별 보상 가중치를 조정할 때 행동 재현 성능 확인
- 변경한 것: `fx_return_1d` 하나로 `w_t = beta + B C_t` 적용
- 고정 조건: E0와 동일한 데이터, 특징, binary 라벨, CPCV 45 splits, seed 42, 200 epochs, L1 0.001
- 데이터: `data/processed/dataset_context_binary.npz`
- 설정 파일: `experiments/2026-06-27/configs/context_e1_fx_1d.yaml`
- 결과 폴더: `runs/context_e1_fx_1d/`
- 주요 결과: 외국인 NLL 0.647269·Macro F1 0.612553·개선 split 53.3%, 기관 0.699462·0.518629·48.9%, 개인 0.683136·0.575280·2.2%
- 해석: 외국인은 과반 split에서 NLL이 개선됐지만 평균 NLL이 E0보다 높아 판정 기준을 통과하지 못함. 기관과 개인도 효과 없음
- 다음 액션: 환율 5일 및 KOSPI 단일 컨텍스트와 비교
