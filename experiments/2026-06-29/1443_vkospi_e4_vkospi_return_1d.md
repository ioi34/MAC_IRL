# vkospi_e4: vkospi_return_1d 컨텍스트

- 날짜/시간: 2026-06-29 14:43
- 목적: VKOSPI 1일 변화율을 컨텍스트로 사용
- 변경한 것: context_names: [vkospi_return_1d]
- 고정 조건: E0과 동일
- 설정 파일: `experiments/2026-06-29/configs/vkospi_e4.yaml`
- 결과 폴더: `runs/vkospi_e4/`

## 주요 결과

| 투자자 | accuracy | macro_f1 | nll |
|--------|----------|----------|-----|
| 외국인 | 0.6303 | 0.6123 | 0.6734 |
| 기관 | 0.5236 | 0.5112 | 0.7146 |
| 개인 | 0.5809 | 0.5637 | 0.6854 |

## 해석

수준값(E3)보다 안정적이나 NLL이 여전히 baseline보다 나쁨.
vkospi 변화율 단독으로는 효과 제한적.
