# 컨텍스트 E2 환율 5일

- 날짜/시간: 2026-06-27 17:19 KST
- 목적: USD/KRW 5일 로그수익률이 투자자별 보상 가중치를 조정할 때 행동 재현 성능 확인
- 변경한 것: `fx_return_5d` 하나로 `w_t = beta + B C_t` 적용
- 고정 조건: E0와 동일한 데이터, 특징, binary 라벨, CPCV 45 splits, seed 42, 200 epochs, L1 0.001
- 데이터: `data/processed/dataset_context_binary.npz`
- 설정 파일: `experiments/2026-06-27/configs/context_e2_fx_5d.yaml`
- 결과 폴더: `runs/context_e2_fx_5d/`
- 주요 결과: 외국인 NLL 0.652348·Macro F1 0.613000·개선 split 37.8%, 기관 0.696820·0.529124·60.0%, 개인 0.683327·0.568191·33.3%
- 해석: 기관만 E0 NLL 0.697315보다 낮고 과반 split에서 개선됐으며 Macro F1도 0.522122에서 0.529124로 상승해 판정 기준 통과. 외국인과 개인은 효과 없음
- 다음 액션: 기관용 컨텍스트 후보로 유지하고 KOSPI 및 통합 모델과 비교
