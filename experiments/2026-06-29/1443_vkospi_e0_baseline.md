# vkospi_e0: baseline (컨텍스트 없음)

- 날짜/시간: 2026-06-29 14:43
- 목적: VKOSPI 컨텍스트 실험 기준선
- 변경한 것: 없음 (컨텍스트 미사용)
- 고정 조건: underwater, herd(5), momentum(20), volatility(Parkinson,20), persist / binary / 973샘플
- 데이터: `data/raw/samsung_macirl_EXTENDED_V2019_2025.csv`
- 설정 파일: `configs/data_vkospi.yaml`, `configs/features_vkospi.yaml`, `experiments/2026-06-29/configs/vkospi_e0.yaml`
- 결과 폴더: `runs/vkospi_e0/`

## 주요 결과

| 투자자 | accuracy | macro_f1 | nll |
|--------|----------|----------|-----|
| 외국인 | 0.6262 | 0.6088 | 0.6569 |
| 기관 | 0.5189 | 0.4951 | 0.7006 |
| 개인 | 0.5800 | 0.5631 | 0.6815 |

## 해석

컨텍스트 없이 5개 보상특징만 사용한 기준 결과.
