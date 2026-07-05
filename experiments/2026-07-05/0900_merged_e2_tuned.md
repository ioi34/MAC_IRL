# 2단계 최종 베이스라인: merged_e2 (HP 재튜닝)

- 날짜/시간: 2026-07-05 09:00
- 목적: 2-컨텍스트(kospi_return_1d + fx_level_z_252) 기준 HP 재튜닝 후 최종 베이스라인 확정
- 변경한 것:
  - `scripts/tune_merged.py` — `--contexts` 인자 추가로 재사용 가능하게 수정
  - `configs/experiment_merged_e2.yaml` 신규 생성 (튜닝된 HP 반영)
- 고정 조건: 6개 피처 / 2개 컨텍스트 / 973샘플 / CPCV 45 split
- 설정 파일: `configs/data_vkospi.yaml`, `configs/features_vkospi.yaml`, `configs/experiment_merged_e2.yaml`
- 결과 폴더: `runs/merged_e2/`

---

## 튜닝 결과 (투자자별 최적 HP)

| 투자자 | epochs | batch_size | lr | lambda_l1 | 탐색 범위 |
|--------|--------|-----------|-----|----------|---------|
| 외국인 | 75 | 256 | 0.001 | 0.0 | ep=[75,100,150], lr=[0.0005,0.001] |
| 기관 | 10 | 2048 | 0.0005 | 0.01 | ep=[10,15,20,25], lr=[0.0003,0.0005,0.00075] |
| 개인 | 20 | 512 | 0.001 | 0.0003 | ep=[10,15,20,25], lr=[0.001,0.0015,0.002] |

---

## 전체 실험 흐름 (accuracy)

| 투자자 | relative_e1 (5피처 1ctx) | pairwise_herd_e1 (6피처 1ctx) | merged_e1 (6피처 2ctx) | **merged_e2 (6피처 2ctx+튜닝)** |
|--------|---------|---------|---------|---------|
| 외국인 | 0.6387 | 0.6225 | 0.6131 | **0.6139** |
| 기관 | 0.5215 | 0.5161 | 0.5098 | **0.5132** |
| 개인 | 0.5980 | 0.6016 | 0.5856 | **0.5867** |

---

## 가중치 결과 (β)

| 투자자 | 피처 | β | 방향 일관성 | 출처 |
|--------|------|--:|----------:|------|
| 외국인 | momentum | +0.1183 | 100% | runs/merged_e2/reward_weights_summary.csv |
| 외국인 | relative | +0.1177 | 100% | 〃 |
| 외국인 | herd_b (개인) | -0.1024 | 100% | 〃 |
| 외국인 | underwater | -0.0486 | 98% | 〃 |
| 외국인 | herd_a (기관) | -0.0308 | 87% | 〃 |
| 외국인 | volatility | +0.0227 | 69% | 〃 |
| 기관 | momentum | -0.0049 | 100% | 〃 |
| 기관 | volatility | +0.0043 | 93% | 〃 |
| 기관 | herd_b (개인) | -0.0030 | 82% | 〃 |
| 기관 | underwater | -0.0028 | 80% | 〃 |
| 기관 | relative | -0.0026 | 78% | 〃 |
| 기관 | herd_a (외국인) | -0.0010 | 69% | 〃 |
| 개인 | relative | -0.0359 | 100% | 〃 |
| 개인 | momentum | -0.0351 | 100% | 〃 |
| 개인 | herd_a (외국인) | -0.0338 | 100% | 〃 |
| 개인 | herd_b (기관) | -0.0336 | 100% | 〃 |
| 개인 | underwater | +0.0282 | 98% | 〃 |
| 개인 | volatility | -0.0162 | 82% | 〃 |

## 컨텍스트 가중치 (B, 100% 일관성)

| 투자자 | 피처 × 컨텍스트 | B | 해석 |
|--------|--------------|--:|------|
| 외국인 | volatility × kospi_return_1d | +0.1029 | KOSPI 강세 시 변동성 프리미엄 확대 |
| 외국인 | momentum × fx_level_z_252 | -0.0847 | **달러 강세 국면에서 추세추종 약화** |
| 기관 | volatility × kospi_return_1d | -0.0049 | KOSPI 강세 시 변동성 민감도 반전 |
| 기관 | volatility × fx_level_z_252 | +0.0049 | 달러 강세 시 변동성 프리미엄 증가 |
| 기관 | herd_b × fx_level_z_252 | +0.0049 | 달러 강세 시 개인 역추종 강화 |
| 기관 | herd_a × fx_level_z_252 | -0.0046 | 달러 강세 시 외국인 역추종 약화 |
| 기관 | underwater × kospi_return_1d | +0.0045 | KOSPI 강세 시 손실 구간 추가매수 |
| 개인 | momentum × fx_level_z_252 | +0.0344 | **달러 강세 국면에서 역추세 성향 강화** |
| 개인 | volatility × kospi_return_1d | -0.0314 | KOSPI 강세 시 변동성 기피 강화 |
| 개인 | underwater × fx_level_z_252 | -0.0282 | 달러 강세 시 물타기 성향 감소 |

---

## 2단계 최종 베이스라인 확정

- 설정: `configs/experiment_merged_e2.yaml`
- 결과: `runs/merged_e2/`
- 보상 피처 6개: underwater, momentum, relative, volatility, herd_a, herd_b
- 모델 컨텍스트 2개: kospi_return_1d, fx_level_z_252

## 근거 구분

- 실제 파일 기반: `runs/merged_e2/cv_metrics_summary.csv`, `reward_weights_summary.csv`, `context_weights_summary.csv`
- 튜닝 결과: `experiments/2026-07-05/tune_merged_kospi_return_1d_fx_level_z_252/all_results.csv`
