# herd pooled vs pairwise 비교

- 날짜/시간: 2026-07-12 18:00
- 목적: 확정 baseline의 herd 표현을 pairwise(herd_a+herd_b)로 둘지, pooled(단일 herd)로 바꿀지 증거로 결정. 쟁점은 herd_a/herd_b −0.82 공선성(가중치 안정성 축).
- 변경한 것: pooled 변형 = underwater·momentum·relative·**herd(window=1)** 4특징. window=1로 둬 pooled가 두 pairwise 신호의 순수 평균이 되게 함(지연 1일 동일) → pooling 효과만 격리.
- 고정 조건: κ=1, CPCV 45 split, split별 train-only 스케일링, 투자자별 HP, seed 42. numpy 재현.
- 데이터: `dataset_continuous_pooled.npz`. config `configs/features_continuous_pooled.yaml`.
- 결과 폴더: `runs/continuous_pooled/`.
- 주요 결과: **pooled 우위.** 성능 약간↑, 특징 −1, 공선성 제거, herd 단일 안정 가중치.

## 성능 비교 (45 CPCV OOS)

| 주체 | pairwise(5특징) 방향/상관 | pooled(4특징) 방향/상관 |
| --- | --- | --- |
| 외국인 | 0.616 / 0.240 | 0.624 / 0.253 |
| 기관 | 0.497 / 0.020 | 0.523 / 0.052 (노이즈) |
| 개인 | 0.595 / 0.233 | 0.599 / 0.243 |

## 가중치 결과 (β, 부호일관성)

| 대상 | 피처 | β(pooled) | 부호일관성 | 비고(pairwise 대비) | 출처 |
| --- | --- | ---: | ---: | --- | --- |
| 외국인 | momentum | +0.047 | 100% | 동일 | runs/continuous_pooled/beta_weights.csv |
| 외국인 | herd | −0.038 | 100% | herd_a −0.015/herd_b −0.045 → 단일화 | 〃 |
| 외국인 | relative | +0.022 | 100% | 동일 | 〃 |
| 외국인 | underwater | +0.001 | 56% | 여전히 약함 | 〃 |
| 개인 | herd | −0.031 | 100% | herd_a −0.030/herd_b −0.025 → 단일화 | 〃 |
| 개인 | underwater | +0.029 | 100% | 동일 | 〃 |
| 개인 | relative | −0.029 | 100% | 동일 | 〃 |
| 개인 | momentum | −0.028 | 100% | 동일 | 〃 |
| 기관 | (전 피처) | ≈0 | 56~100% | 무신호 유지 | 〃 |

- 해석: pooled는 성능 손실 없이(오히려 소폭↑) herd_a/herd_b 공선성을 제거하고 해석을 "군집 추종 1개 가중치"로 단순화. 가중치 안정성 축에 유리 → 채택 권장.
- 다음 액션: pooled 4특징을 새 확정 baseline으로 채택 후, 추가 후보(VKOSPI 컨텍스트·단기모멘텀·FX레벨 직접피처) 하나씩 검증(3단계). underwater(외국인 56%)는 추가 검증 병행.
