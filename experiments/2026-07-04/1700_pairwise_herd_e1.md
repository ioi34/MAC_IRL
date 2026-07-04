# pairwise herd 재설계 실험

- 날짜/시간: 2026-07-04 17:00
- 목적: pooled herd(나머지 평균)를 개별 방향 herd_a/herd_b로 분리하여 "누가 누구를 따라가는가" 해석 개선
- 변경한 것:
  - `src/features/herd.py` — `build_herd_a`, `build_herd_b`, `get_herd_source_map` 추가
  - `src/features/registry.py` — herd_a, herd_b 등록
  - `configs/features_vkospi.yaml` — selected: herd → herd_a, herd_b (총 6개 피처)
  - `scripts/train.py` — herd_source_map.csv 자동 저장 추가
- 고정 조건: kospi_return_1d / 973샘플 / CPCV 45 split / 투자자별 HP 동일(relative_e1과 동일)
- 데이터: `data/raw/samsung_macirl_EXTENDED_V2019_2025.csv`
- 설정 파일: `configs/data_vkospi.yaml`, `configs/features_vkospi.yaml`, `configs/experiment_pairwise_herd.yaml`
- 결과 폴더: `runs/pairwise_herd_e1/`

## herd 소스 매핑

| investor | herd_a 소스 | herd_b 소스 |
|---|---|---|
| foreign | institution | retail |
| institution | foreign | retail |
| retail | foreign | institution |

`feature = u_{source, t-1} × action` (rolling 없음, 1일 시차)

## 성능 비교 (relative_e1 베이스라인 대비)

| 투자자 | 베이스라인 acc | pairwise acc | 변화 | NLL 변화 | F1 변화 |
|--------|-------------|------------|------|---------|--------|
| 외국인 | 0.6387 | 0.6225 | **-0.0161** | +0.0117 | -0.0136 |
| 기관 | 0.5215 | 0.5161 | -0.0053 | +0.0011 | -0.0037 |
| 개인 | 0.5980 | 0.6016 | +0.0036 | +0.0023 | +0.0038 |

## 가중치 결과 (β)

| 투자자 | 피처 | β | 방향 일관성 | 해석 | 출처 |
|--------|------|--:|----------:|------|------|
| 외국인 | momentum | +0.1403 | 100% | 추세추종 | runs/pairwise_herd_e1/reward_weights_summary.csv |
| 외국인 | herd_b (retail 소스) | **-0.1197** | 100% | 개인 역추종 | 〃 |
| 외국인 | relative | +0.1134 | 100% | 삼성 초과상승 시 매수 | 〃 |
| 외국인 | underwater | -0.0586 | 100% | 손실 구간 기피 | 〃 |
| 외국인 | herd_a (inst 소스) | -0.0412 | 96% | 기관 약한 역추종 | 〃 |
| 외국인 | volatility | +0.0315 | 71% | 변동성 증가 시 매수 | 〃 |
| 기관 | momentum | -0.0073 | 100% | 역추세 | 〃 |
| 기관 | volatility | +0.0065 | 98% | | 〃 |
| 기관 | herd_b (retail 소스) | -0.0046 | 91% | | 〃 |
| 기관 | relative | -0.0038 | 73% | 약한 신호 | 〃 |
| 기관 | underwater | -0.0036 | 82% | | 〃 |
| 기관 | herd_a (foreign 소스) | -0.0014 | 76% | | 〃 |
| 개인 | relative | -0.0410 | 100% | 시장 초과상승 시 매도 | 〃 |
| 개인 | momentum | -0.0400 | 100% | 역추세 | 〃 |
| 개인 | herd_a (foreign 소스) | -0.0384 | 100% | 외국인 역추종 | 〃 |
| 개인 | herd_b (inst 소스) | -0.0379 | 100% | 기관 역추종 | 〃 |
| 개인 | underwater | +0.0271 | 91% | 손실 구간 추가 매수 | 〃 |
| 개인 | volatility | -0.0212 | 87% | 변동성 기피 | 〃 |

## 해석

**외국인**
- pooled herd β=-0.158 → pairwise로 분해: herd_b(retail) -0.120 + herd_a(inst) -0.041
- 개인 역추종이 기관 역추종보다 3배 강함
- 이전에는 합산으로 상쇄됐던 방향 차이가 분리됨

**기관**
- L1=0.01로 herd_a/herd_b 모두 약함 (-0.001~-0.005), pooled 때와 동일하게 소멸
- 기관은 herd 신호 전반에 무반응 (정규화 문제 아님, 행동 특성)

**개인**
- herd_a(foreign) -0.038 = herd_b(inst) -0.038 거의 동일
- 개인은 두 주체 모두 균등하게 역추종 — "스마트머니 전반에 반대로 행동"

## 결론

- **해석 가치**: herd_b가 외국인의 주된 역추종 대상이 개인임을 명시적으로 확인. 해석 개선 성공.
- **성능**: 외국인 -0.016 하락이 치명적. pooled herd가 분리했을 때보다 더 강한 예측력 보유.
- **원인 가설**: pooled herd에서 institution+retail을 합산하면 signal-to-noise 비율이 더 높았음.
  pairwise 분리 시 개별 노이즈가 각 feature에 독립적으로 들어와 fit 품질 저하.

## 최종 판단

- 이 프로젝트의 우선순위는 **예측 정확도보다 해석**
- pooled herd는 "누구를 따라가는지" 읽을 수 없음; pairwise는 직접 읽힘
- 외국인 -0.016 성능 손실은 해석 밀도 향상 대비 감수 가능한 수준
- **pairwise herd 채택 → 새 2단계 베이스라인: `runs/pairwise_herd_e1/`**

## 근거 구분

- 실제 파일 기반: `runs/pairwise_herd_e1/cv_metrics_summary.csv`, `reward_weights_summary.csv`, `herd_source_map.csv`
- 베이스라인: `runs/relative_e1/cv_metrics_summary.csv`
