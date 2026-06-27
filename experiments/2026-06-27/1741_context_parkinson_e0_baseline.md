# Parkinson 컨텍스트 E0 기준선

- 날짜/시간: 2026-06-27 17:41 KST
- 목적: Parkinson 변동성을 포함한 새 표준 보상특징의 고정 가중치 기준선 확보
- 변경한 것: 기존 underwater + herd + momentum에 Parkinson volatility(window=20) 추가
- 고정 조건: binary rolling median, 973개 표본, CPCV 45 splits, seed 42, 200 epochs, L1 0.001
- 데이터: `data/processed/dataset_context_binary.npz`
- 설정 파일: `configs/data_context.yaml`, `configs/features_context.yaml`, `experiments/2026-06-27/configs/context_parkinson_e0_baseline.yaml`
- 결과 폴더: `runs/context_parkinson_e0_baseline/`
- 주요 결과: 외국인 NLL 0.656193·Macro F1 0.610206, 기관 0.697890·0.486168, 개인 0.682851·0.564459
- 해석: 이후 컨텍스트 실험은 이 Parkinson 기준선과만 비교
- 다음 액션: 동일 데이터와 split으로 E1~E5 실행
