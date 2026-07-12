# 실험용 데이터 통합 (vkospi 추가) 및 canonical 설정

- 날짜/시간: 2026-07-12 16:00
- 목적: 실험 기본 데이터셋에 VKOSPI를 포함시키고, 이를 실험이 실제로 읽는 canonical 데이터로 설정.
- 변경한 것:
  - 베이스(2019–2025, semi 포함 v2)에 **VKOSPI 컬럼(`vkospi`)** 병합.
  - canonical 파일 2곳을 통합본으로 덮어씀: 루트 `samsung_macirl_EXTENDED_2019_2025.csv`, `data/raw/samsung_macirl_EXTENDED_2019_2025.csv`.
  - 기본 실험 파이프라인(`train.py` 기본 `--data-config configs/data_context.yaml`)이 루트 파일을 읽으므로 별도 config 수정 없이 vkospi가 실험 데이터에 반영됨.
- 고정 조건: 행 수 1,719일, 날짜·기존 컬럼 불변. VKOSPI는 raw 컬럼으로만 추가(모델 피처 선택은 features config 몫 — 자동 사용 아님).
- 데이터:
  - VKOSPI 소스: KRX xlsx 7개(2019–2025, `data_2138/2206/2224/2240/2409/2428/2443_20260627.xlsx`)의 종가.
  - 검증: 기존 `samsung_macirl_EXTENDED_V2019_2025.csv`의 vkospi와 **최대차이 0.0 · 불일치 0일** → 소스 동일 확인. 결측 0. 범위 11.7~69.2(2020-03 코로나 69.2 정점).
- 설정 파일: `configs/data_context.yaml`(raw_daily = `samsung_macirl_EXTENDED_2019_2025.csv`) — 수정 불필요, 파일 교체로 반영.
- 결과 폴더: 데이터 `MAC_IRL/samsung_macirl_EXTENDED_2019_2025.csv`, `MAC_IRL/data/raw/…`, 사본 `outputs/…{csv,xlsx}`.
- 주요 결과: 통합본 25열(기존 23 + `semi` + `vkospi`), 1,719일, 결측 0. 실험 기본 파이프라인이 vkospi 포함 데이터를 읽도록 설정 완료.

## 가중치 결과

| 대상 | 피처/가중치 | 값 | 변동성/일관성 | 출처 |
| --- | --- | ---: | ---: | --- |
| — | — | 미산출 | — | 데이터 통합·설정 작업이며 모델 재학습 미실행 |

- 해석: vkospi는 이제 실험 데이터에 존재하나, 실제 모델 반영은 features/contexts config에서 선택해야 발동. 즉 "데이터 준비 완료" 단계.
- 다음 액션: vkospi를 컨텍스트(위험회피 국면) 또는 직접 피처로 넣는 features config 구성 후 외국인·개인 acc/NLL ablation. 앞서 스크리닝의 단기모멘텀·FX레벨 후보와 함께 비교.
