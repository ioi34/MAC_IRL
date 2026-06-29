# persist 제외 VKOSPI 컨텍스트 실험 종합

- 날짜/시간: 2026-06-29 15:37
- 목적: persist 제외 후 VKOSPI 컨텍스트 영향 비교 (persist 포함 실험과 대조)
- 변경한 것: 보상특징에서 persist 제거 (underwater, herd, momentum, volatility만 사용)
- 고정 조건: binary / 973샘플 (2022-01-06 ~ 2025-12-29) / CPCV 동일
- 데이터: `data/raw/samsung_macirl_EXTENDED_V2019_2025.csv`
- 설정 파일: `configs/data_vkospi.yaml`, `configs/features_no_persist_vkospi.yaml`
- 결과 폴더: `runs/no_persist_e0/` ~ `runs/no_persist_e7/`

## 전체 결과 비교

| 실험 | 컨텍스트 | 외국인 acc | 기관 acc | 개인 acc | 외국인 NLL |
|------|---------|-----------|---------|---------|-----------|
| E0 | baseline | 0.6279 | 0.5163 | 0.5824 | 0.6562 |
| E1 | fx_return_1d | 0.6206 | 0.5272 | 0.5820 | 0.6561 |
| E2 | kospi_return_1d | **0.6391** | 0.5284 | 0.5886 | **0.6547** |
| E3 | vkospi_1d | 0.6364 | 0.5243 | 0.5850 | 0.6643 |
| E4 | vkospi_return_1d | 0.6326 | 0.5223 | 0.5782 | 0.6695 |
| E5 | vkospi_return_5d | 0.6342 | 0.5203 | 0.5738 | 0.6599 |
| E6 | kospi_1d + vkospi_return_1d | 0.6295 | **0.5331** | **0.5893** | 0.6752 |
| E7 | fx_1d + kospi_1d + vkospi_return_1d | 0.6233 | 0.5400 | 0.5894 | 0.6781 |

## persist 포함 vs 미포함 비교 (주요 실험)

| 실험 | | 외국인 acc | 기관 acc | 개인 acc | 외국인 NLL |
|------|--|-----------|---------|---------|-----------|
| E0 baseline | persist 포함 | 0.6262 | 0.5189 | 0.5800 | 0.6569 |
| E0 baseline | **persist 제외** | 0.6279 | 0.5163 | 0.5824 | 0.6562 |
| E2 kospi_1d | persist 포함 | 0.6315 | 0.5281 | 0.5887 | 0.6526 |
| E2 kospi_1d | **persist 제외** | **0.6391** | 0.5284 | 0.5886 | 0.6547 |
| E3 vkospi_1d | persist 포함 | 0.6373 | 0.5125 | 0.5798 | 0.6667 |
| E3 vkospi_1d | **persist 제외** | 0.6364 | 0.5243 | 0.5850 | 0.6643 |

## 해석

- **kospi_return_1d(E2)**: persist 제외 시 외국인 accuracy 0.6391로 전체 실험 최고. NLL도 0.6547로 양호
- **vkospi_1d(E3)**: persist 포함/제외 결과가 유사. 외국인에만 유효한 패턴 유지
- **E6·E7 조합**: 기관·개인 accuracy는 높지만 외국인 NLL 악화 패턴도 동일하게 유지
- **persist 영향**: 전반적으로 미미. 외국인 accuracy에서 persist 제외가 약간 유리하나 NLL 기준으로는 포함이 더 나음

## 결론

persist 유무와 관계없이 **kospi_return_1d 단독이 가장 균형 잡힌 컨텍스트**임을 재확인.
vkospi는 외국인 accuracy 향상에 일정 기여하나 NLL 악화를 동반해 단독 채택은 신중히 검토 필요.
persist 제외가 외국인 accuracy를 소폭 높이나 NLL 기준으로는 포함이 더 정확한 예측을 함.

## 다음 액션

1. kospi_return_1d 컨텍스트를 메인 후보로 확정 검토
2. vkospi_1d의 외국인 accuracy 향상이 실제 신호인지 과적합인지 추가 검증 필요
3. 동료와 결과 공유

## 근거 구분

- 실제 파일 기반: `runs/no_persist_e0/` ~ `runs/no_persist_e7/` cv_metrics.csv
