# 통합 실험: pairwise herd + fx_level_z_252 컨텍스트

- 날짜/시간: 2026-07-04 17:48
- 목적: tony(pairwise herd, relative) + yumin(fx_level_z_252 컨텍스트) 결과 통합
- 변경한 것:
  - `configs/features_vkospi.yaml` — contexts에 `fx_level_z_252` 추가 (총 6개 컨텍스트 저장)
  - `configs/experiment_merged_e1.yaml` 신규 생성 — 모델 컨텍스트: kospi_return_1d + fx_level_z_252
- 고정 조건: 6개 피처 / 973샘플 / CPCV 45 split / 투자자별 HP 동일(pairwise_herd_e1과 동일)
- 설정 파일: `configs/data_vkospi.yaml`, `configs/features_vkospi.yaml`, `configs/experiment_merged_e1.yaml`
- 결과 폴더: `runs/merged_e1/`

---

## 현재 보상 특징 및 컨텍스트 상태

### 보상 특징 (6개)

| 피처 | 정의 | 목적 |
|------|------|------|
| underwater | 평균매입단가 대비 손실 비율 × action | 손실회피 / 물타기 |
| momentum | 삼성 20일 log수익률 × action | 추세추종 / 역추세 |
| relative | (삼성 − KOSPI200) 20일 수익률 × action | 벤치마크 대비 초과성과 |
| volatility | Parkinson 20일 변동성 × action (방향형) | 변동성 민감도 |
| herd_a | 자기 제외 첫 번째 주체 전일 순매수비율 × action | 개별 군집 추종/역추종 |
| herd_b | 자기 제외 두 번째 주체 전일 순매수비율 × action | 개별 군집 추종/역추종 |

**herd 소스 매핑**

| 투자자 | herd_a 소스 | herd_b 소스 |
|--------|-----------|-----------|
| 외국인 | 기관 | 개인 |
| 기관 | 외국인 | 개인 |
| 개인 | 외국인 | 기관 |

### 컨텍스트 (모델에 사용: 2개)

| 컨텍스트 | 정의 | 출처 |
|---------|------|------|
| kospi_return_1d | KOSPI200 당일 수익률 | 시장 방향 |
| fx_level_z_252 | log(USDKRW)의 252일 rolling z-score | 환율 수준 (달러 강세/약세 국면) |

*데이터셋에는 추가로 fx_return_1d, vkospi_1d, vkospi_return_1d/5d도 저장됨 (향후 컨텍스트 실험용)*

---

## 성능 비교 (실험 흐름)

| 투자자 | relative_e1 | pairwise_herd_e1 | merged_e1 | 누적 변화 |
|--------|------------|-----------------|-----------|---------|
| 외국인 | 0.6387 | 0.6225 | 0.6131 | -0.026 |
| 기관 | 0.5215 | 0.5161 | 0.5098 | -0.012 |
| 개인 | 0.5980 | 0.6016 | 0.5856 | -0.012 |

*성능 하락 원인: HP가 단일 컨텍스트(kospi_return_1d) 기준으로 튜닝된 상태에서 컨텍스트 2개로 확장*

---

## 가중치 결과 (β)

| 투자자 | 피처 | β | 방향 일관성 | 출처 |
|--------|------|--:|----------:|------|
| 외국인 | relative | +0.1292 | 100% | runs/merged_e1/reward_weights_summary.csv |
| 외국인 | momentum | +0.1280 | 100% | 〃 |
| 외국인 | herd_b (개인) | -0.1096 | 100% | 〃 |
| 외국인 | underwater | -0.0485 | 93% | 〃 |
| 외국인 | herd_a (기관) | -0.0388 | 93% | 〃 |
| 외국인 | volatility | +0.0182 | 60% | 〃 |
| 기관 | momentum | -0.0073 | 100% | 〃 |
| 기관 | volatility | +0.0063 | 93% | 〃 |
| 기관 | herd_b (개인) | -0.0046 | 87% | 〃 |
| 기관 | underwater | -0.0040 | 82% | 〃 |
| 기관 | relative | -0.0038 | 76% | 〃 |
| 기관 | herd_a (외국인) | -0.0014 | 62% | 〃 |
| 개인 | relative | -0.0404 | 100% | 〃 |
| 개인 | momentum | -0.0394 | 100% | 〃 |
| 개인 | herd_b (기관) | -0.0380 | 100% | 〃 |
| 개인 | herd_a (외국인) | -0.0379 | 100% | 〃 |
| 개인 | underwater | +0.0317 | 100% | 〃 |
| 개인 | volatility | -0.0183 | 84% | 〃 |

## 컨텍스트 가중치 결과 (B, 100% 일관성 중심)

| 투자자 | 피처 × 컨텍스트 | B | 방향 일관성 | 해석 |
|--------|--------------|--:|----------:|------|
| 외국인 | volatility × kospi_return_1d | +0.1152 | 100% | KOSPI 강세 시 변동성 프리미엄 확대 |
| 외국인 | momentum × fx_level_z_252 | -0.0971 | 100% | **달러 강세 국면에서 추세추종 약화** |
| 외국인 | underwater × fx_level_z_252 | -0.0567 | 96% | 달러 강세 시 손실 구간 매도 성향 감소 |
| 기관 | volatility × kospi_return_1d | -0.0074 | 100% | KOSPI 강세 시 변동성 민감도 반전 |
| 기관 | volatility × fx_level_z_252 | +0.0073 | 100% | 달러 강세 시 변동성 프리미엄 증가 |
| 기관 | herd_b × fx_level_z_252 | +0.0072 | 100% | 달러 강세 시 개인 역추종 강화 |
| 개인 | momentum × fx_level_z_252 | +0.0387 | 100% | **달러 강세 국면에서 역추세 성향 강화** |
| 개인 | volatility × kospi_return_1d | -0.0354 | 100% | KOSPI 강세 시 변동성 기피 강화 |
| 개인 | underwater × fx_level_z_252 | -0.0318 | 100% | 달러 강세 시 물타기 성향 감소 |
| 개인 | relative × fx_level_z_252 | +0.0286 | 98% | 달러 강세 시 벤치마크 역추세 성향 강화 |

---

## 해석 요약

**외국인**
- 추세추종(momentum β=+0.128) + 벤치마크 초과(relative β=+0.129) 주도
- 달러 강세 국면(fx_level_z_252 높을 때)에서 추세추종 성향 약화 (B=-0.097, 100%) → 환율 리스크 헤징 행동 해석 가능

**기관**
- L1=0.01로 모든 피처 신호 약함. 컨텍스트 효과도 미미하나 volatility × 두 컨텍스트 모두 100% 일관성.

**개인**
- 역추세(momentum β=-0.039) + 두 주체 균등 역추종(herd_a≈herd_b≈-0.038) + 벤치마크 역추종(relative β=-0.040)
- 달러 강세 국면에서 역추세 성향 더 강화 (B=+0.039, 100%)

---

## 다음 액션

1. **HP 재튜닝**: 2-컨텍스트 기준으로 epochs/lr/lambda_l1 재탐색 → 성능 회복 가능성
2. **컨텍스트 단독 실험**: fx_level_z_252만 단독 사용 vs kospi_return_1d 단독 비교
3. 현재 상태로 2단계 베이스라인 확정 후 3단계 설계

## 근거 구분

- 실제 파일 기반: `runs/merged_e1/cv_metrics_summary.csv`, `reward_weights_summary.csv`, `context_weights_summary.csv`
- 비교 기준: `runs/relative_e1/`, `runs/pairwise_herd_e1/`
