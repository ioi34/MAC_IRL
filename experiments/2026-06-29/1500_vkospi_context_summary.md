# VKOSPI 컨텍스트 실험 종합

- 날짜/시간: 2026-06-29 15:00
- 목적: VKOSPI를 환율·KOSPI200과 함께 컨텍스트 후보로 비교
- 고정 조건: underwater, herd(5), momentum(20), volatility(Parkinson,20), persist / binary / 973샘플
- 데이터: `data/raw/samsung_macirl_EXTENDED_V2019_2025.csv`
- 설정 파일: `configs/data_vkospi.yaml`, `configs/features_vkospi.yaml`
- 결과 폴더: `runs/vkospi_e0/` ~ `runs/vkospi_e7/`

## 전체 결과 비교

| 실험 | 컨텍스트 | 외국인 acc | 기관 acc | 개인 acc | 외국인 NLL |
|------|---------|-----------|---------|---------|-----------|
| E0 | baseline | 0.6262 | 0.5189 | 0.5800 | 0.6569 |
| E1 | fx_return_1d | 0.6214 | 0.5176 | 0.5796 | 0.6576 |
| E2 | kospi_return_1d | 0.6315 | 0.5281 | 0.5887 | **0.6526** |
| E3 | vkospi_1d | **0.6373** | 0.5125 | 0.5798 | 0.6667 |
| E4 | vkospi_return_1d | 0.6303 | 0.5236 | 0.5809 | 0.6734 |
| E5 | vkospi_return_5d | 0.6304 | 0.5194 | 0.5860 | 0.6644 |
| E6 | kospi_1d + vkospi_return_1d | 0.6250 | **0.5312** | **0.5930** | 0.6764 |
| E7 | fx_1d + kospi_1d + vkospi_return_1d | 0.6212 | 0.5385 | 0.5918 | 0.6795 |

## 해석

- **kospi_return_1d(E2)**: NLL 기준 최고, 전 투자자 균형 잡힌 개선 → 가장 안정적인 컨텍스트
- **vkospi_1d(E3)**: 외국인 accuracy 최고(0.6373)이나 NLL 악화, 기관·개인 하락 → 과적합 가능성
- **E6 조합**: 기관·개인 accuracy 최고이나 외국인 NLL 악화
- **E7 전체 조합**: 기관 accuracy 0.5385로 최고이나 외국인 NLL 0.6795로 가장 나쁨. 컨텍스트를 늘릴수록 외국인 예측 신뢰도 하락

**컨텍스트 가중치 주요 발견 (E7):**
- 외국인: kospi_return_1d가 underwater(-0.171), herd(-0.199), persist(-0.180)에 일관되게 음수 → 시장 상승 시 외국인이 매도 압력 증가
- 기관·개인: vkospi_return_1d가 herd(+0.124, +0.122), persist(+0.092, +0.086)에 양수 → 변동성 상승 시 군집·자기지속 강화

## 결론

컨텍스트를 늘릴수록 기관·개인 accuracy는 오르지만 외국인 NLL이 악화됨.
외국인에는 kospi_return_1d 단독이 가장 균형 잡힌 컨텍스트.
기관·개인에는 vkospi_return_1d 추가가 유효.

## 다음 액션

1. vkospi_1d가 외국인에만 강한 이유 분석 (변동성 수준과 외국인 행동 패턴 관계)
2. 투자자별 최적 컨텍스트 분리 모델 검토
3. 동료와 결과 공유

## 근거 구분

- 실제 파일 기반: `runs/vkospi_e0/` ~ `runs/vkospi_e7/` cv_metrics.csv
