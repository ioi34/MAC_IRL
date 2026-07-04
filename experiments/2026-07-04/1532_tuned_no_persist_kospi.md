# 투자자별 튜닝 하이퍼파라미터 + persist 제외 실험

- 날짜/시간: 2026-07-04 15:32
- 목적: 유민의 투자자별 최적 하이퍼파라미터를 tony 브랜치에 반영하고, persist 보상특징 제외 결정을 최종 적용
- 변경한 것:
  - `configs/features_vkospi.yaml`에서 persist 제거 (4개 피처로 확정)
  - `configs/experiment_tuned_kospi.yaml` 생성 (투자자별 개별 하이퍼파라미터 적용)
- 고정 조건: kospi_return_1d 컨텍스트 / 973샘플 (2022-01-06 ~ 2025-12-29) / CPCV 10-fold/2-test-fold (45 split) / purge 1 / embargo 5 / seed 42
- 데이터: `data/raw/samsung_macirl_EXTENDED_V2019_2025.csv`
- 설정 파일: `configs/data_vkospi.yaml`, `configs/features_vkospi.yaml`, `configs/experiment_tuned_kospi.yaml`
- 결과 폴더: `runs/tuned_kospi_e2/`

## 하이퍼파라미터 (유민 튜닝 결과 반영)

| 투자자 | epochs | batch_size | learning_rate | lambda_l1 |
|--------|--------|-----------|---------------|-----------|
| 외국인 | 100 | 256 | 0.001 | 0.0 |
| 기관 | 15 | 2048 | 0.0005 | 0.01 |
| 개인 | 15 | 512 | 0.0015 | 0.0003 |

## 성능 결과

| 투자자 | accuracy | macro_f1 | nll |
|--------|---------|---------|-----|
| 외국인 | **0.6399** | 0.6234 | 0.6501 |
| 기관 | 0.5343 | 0.5190 | 0.6923 |
| 개인 | 0.5905 | 0.5787 | 0.6817 |

## persist 포함 vs 제외 비교 (튜닝 하이퍼파라미터 기준)

| 투자자 | persist 포함 | persist 제외 | 변화 |
|--------|------------|------------|------|
| 외국인 accuracy | 0.6379 | **0.6399** | +0.002 |
| 기관 accuracy | 0.5457 | 0.5343 | -0.011 |
| 개인 accuracy | 0.6008 | 0.5905 | -0.010 |
| 외국인 NLL | 0.6498 | 0.6501 | +0.000 |

## 보상 가중치 요약

| 투자자 | 피처 | 방향 | 일관성 | 해석 |
|--------|------|------|--------|------|
| 외국인 | herd | 음 | 100% | 군집 반대 방향 |
| 외국인 | momentum | 양 | 100% | 추세 추종 |
| 외국인 | underwater | 음 | 100% | 손실 구간 회피 |
| 외국인 | volatility | 양 | 73% | 불안정 (방향 일관성 낮음) |
| 개인 | underwater | 양 | 91% | 손실 구간 매수 (물타기) |
| 개인 | herd/momentum | 음 | 100% | 역추세 성향 |

## 해석

- persist 제거 시 외국인 accuracy 소폭 개선, 기관·개인 소폭 하락
- 차이가 작고 방향이 투자자마다 엇갈리므로 persist 제외로 최종 결정
- 기관의 lambda_l1=0.01(강한 정규화)로 인해 가중치가 작게 수렴하는 경향

## 결론

- persist 제외 + 투자자별 튜닝 하이퍼파라미터를 현재 2단계 베이스라인으로 확정
- 보상특징: underwater, herd, momentum, volatility (4개)
- 컨텍스트: kospi_return_1d

## 다음 액션

1. 현재 설정을 2단계 베이스라인으로 확정
2. 3단계(컨텍스트 + 상태 기반 신경망) 구조 설계 시작

## 근거 구분

- 실제 파일 기반: `runs/tuned_kospi_e2/cv_metrics_summary.csv`, `reward_weights_summary.csv`
