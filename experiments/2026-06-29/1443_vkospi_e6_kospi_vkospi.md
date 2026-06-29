# vkospi_e6: kospi_return_1d + vkospi_return_1d 컨텍스트

- 날짜/시간: 2026-06-29 14:43
- 목적: KOSPI200과 VKOSPI 변화율 조합
- 변경한 것: context_names: [kospi_return_1d, vkospi_return_1d]
- 고정 조건: E0과 동일
- 설정 파일: `experiments/2026-06-29/configs/vkospi_e6.yaml`
- 결과 폴더: `runs/vkospi_e6/`

## 주요 결과

| 투자자 | accuracy | macro_f1 | nll |
|--------|----------|----------|-----|
| 외국인 | 0.6250 | 0.6085 | 0.6764 |
| 기관 | 0.5312 | 0.5232 | 0.7152 |
| 개인 | 0.5930 | 0.5782 | 0.6917 |

## 해석

기관 accuracy 0.5312, 개인 accuracy 0.5930으로 전체 실험 중 최고.
단, 외국인 NLL 0.6764로 악화. 조합이 기관·개인에 유리하지만 외국인에는 불리.
