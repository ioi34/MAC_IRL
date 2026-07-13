# 확정 베이스라인 — volatility 제거 (5특징)

- 날짜/시간: 2026-07-12 17:45
- 목적: ablation 결과(volatility가 외국인 성능을 방해)에 따라 volatility를 제거한 **확정 연속 베이스라인** 수립.
- 변경한 것: 특징 6→5 (underwater·momentum·relative·herd_a·herd_b). config: `configs/features_continuous_confirmed.yaml`, processed: `dataset_continuous_confirmed.npz`.
- 고정 조건: κ=1, CPCV 45 split, split별 train-only 스케일링, 투자자별 HP, seed 42. numpy 재현.
- 결과 폴더: `runs/continuous_confirmed/` (cv_metrics.csv, beta_weights.csv).
- 주요 결과: 외국인 방향 0.587→0.616·강도상관 0.208→0.240·RMSE 0.253→0.249로 개선. 개인 불변(0.595/0.233). 기관 노이즈(0.497/0.020).

## 성능 비교 (45 CPCV OOS)

| 주체 | 방향정확도 6→5 | 강도상관 6→5 | RMSE 6→5 |
| --- | --- | --- | --- |
| 외국인 | 0.587 → 0.616 | 0.208 → 0.240 | 0.253 → 0.249 |
| 기관 | 0.510 → 0.497 | 0.055 → 0.020 | 0.125 → 0.125 |
| 개인 | 0.594 → 0.595 | 0.232 → 0.233 | 0.319 → 0.320 |

## 가중치 결과 (β 평균, 부호일관성)

| 대상 | 피처 | β | 부호일관성 | 출처 |
| --- | --- | ---: | ---: | --- |
| 외국인 | momentum | +0.047 | 100% | runs/continuous_confirmed/beta_weights.csv |
| 외국인 | herd_b | −0.045 | 100% | 〃 |
| 외국인 | relative | +0.022 | 100% | 〃 |
| 외국인 | herd_a | −0.015 | 98% | 〃 |
| 외국인 | underwater | +0.003 | 67% | 〃 (약함) |
| 기관 | (전 피처) | ≈0 | 60~93% | 〃 |
| 개인 | herd_a | −0.030 | 100% | 〃 |
| 개인 | relative | −0.029 | 100% | 〃 |
| 개인 | underwater | +0.029 | 100% | 〃 |
| 개인 | momentum | −0.028 | 100% | 〃 |
| 개인 | herd_b | −0.025 | 100% | 〃 |

- 해석: volatility 제거는 순이득. 외국인 개선·개인 유지·기관 무영향. underwater는 외국인에서 여전히 약(67%)하나 개인엔 100%로 기여 → 유지하되 다음 검토.
- 다음 액션: (1) herd pooled vs pairwise 안정성 비교, (2) 추가 후보(VKOSPI 컨텍스트·단기모멘텀·FX레벨 직접피처) 하나씩 검증.
