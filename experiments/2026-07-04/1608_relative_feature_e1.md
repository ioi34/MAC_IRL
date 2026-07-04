# relative 보상특징 추가 실험

- 날짜/시간: 2026-07-04 16:08
- 목적: 삼성전자-KOSPI200 상대수익률 feature(relative) 추가 시 성능 및 가중치 변화 확인
- 변경한 것:
  - `src/features/relative.py` 신규 구현
  - `src/features/registry.py` 등록 (build_persist 제거 포함)
  - `configs/features_vkospi.yaml` — relative 추가 (5개 피처)
- 고정 조건: kospi_return_1d 컨텍스트 / 973샘플 / CPCV 45 split / 투자자별 튜닝 HP
- 데이터: `data/raw/samsung_macirl_EXTENDED_V2019_2025.csv`
- 설정 파일: `configs/data_vkospi.yaml`, `configs/features_vkospi.yaml`, `configs/experiment_relative.yaml`
- 결과 폴더: `runs/relative_e1/`

## relative 피처 정의

```
relative = (삼성 20일 log 수익률) − (KOSPI200 20일 누적 log 수익률)
feature  = relative × action
```

- 공선성 사전 확인: momentum vs relative 상관계수 0.616, VIF 1.61 → 문제없음

## 성능 비교

| 투자자 | 베이스라인 acc | relative 추가 acc | 변화 | NLL 변화 | F1 변화 |
|--------|-------------|-----------------|------|---------|--------|
| 외국인 | 0.6399 | 0.6387 | -0.0013 | +0.0047 | +0.0003 |
| 기관 | 0.5343 | 0.5215 | **-0.0128** | +0.0004 | -0.0077 |
| 개인 | 0.5905 | **0.5980** | +0.0074 | -0.0035 | +0.0076 |

## 가중치 결과

| 투자자 | 피처 | β | 방향 일관성 | 출처 |
|--------|------|--:|----------:|------|
| 외국인 | herd | -0.1584 | 100% | runs/relative_e1/reward_weights_summary.csv |
| 외국인 | momentum | +0.1157 | 100% | 〃 |
| 외국인 | relative | **+0.0938** | **100%** | 〃 |
| 외국인 | underwater | -0.0715 | 100% | 〃 |
| 외국인 | volatility | +0.0329 | 76% | 〃 |
| 기관 | herd | -0.0062 | 100% | 〃 |
| 기관 | momentum | -0.0073 | 100% | 〃 |
| 기관 | volatility | +0.0065 | 96% | 〃 |
| 기관 | relative | -0.0037 | 73% | 〃 |
| 개인 | herd | -0.0413 | 100% | 〃 |
| 개인 | momentum | -0.0397 | 100% | 〃 |
| 개인 | relative | **-0.0408** | **100%** | 〃 |
| 개인 | underwater | +0.0271 | 93% | 〃 |
| 개인 | volatility | -0.0213 | 87% | 〃 |

## 해석

**외국인**
- relative β=+0.094(100%): 시장 대비 삼성이 더 오를 때 매수. 절대 모멘텀(+0.116)과 방향 일치하지만 독립적 기여.
- momentum β 감소(+0.144 → +0.116): L1 정규화로 momentum-relative 중복분이 relative로 분산됨. 예상된 현상.

**기관**
- relative β=-0.004(73%): 약하고 일관성 낮음. L1=0.01 강한 정규화로 거의 0에 수렴.
- 성능 -0.013 하락: relative가 기관 행동 설명에 기여하지 못하면서 노이즈로 작용.

**개인**
- relative β=-0.041(100%): 시장 대비 삼성이 초과 상승할 때 매도 (역추세 성향과 일치).
- acc +0.007, F1 +0.008 개선: relative가 개인 행동 설명에 유효.

## 결론

- 외국인·개인에게 relative는 유의미 (β 일관성 100%, 명확한 경제적 해석)
- 기관에게는 L1 정규화에 의해 무력화 → 기관 성능 소폭 하락
- 전체 평균 기준으로 외국인은 거의 유지, 개인은 개선, 기관은 악화
- 현재 설정 유지 여부는 해석 목적 우선순위에 달림

## 다음 액션

1. relative 추가를 최종 베이스라인으로 채택할지 결정
2. 기관의 lambda_l1을 낮춰 relative가 살아남도록 조정하는 실험 (선택적)
3. herd 재설계 (pooled → 개별 주체별) 검토

## 근거 구분

- 실제 파일 기반: `runs/relative_e1/cv_metrics_summary.csv`, `reward_weights_summary.csv`, `context_weights_summary.csv`
- 베이스라인: `runs/tuned_kospi_e2/`
