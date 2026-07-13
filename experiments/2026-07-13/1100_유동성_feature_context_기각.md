# 유동성(Amihud) 특징 — feature·context 둘 다 검증 후 기각

- 날짜/시간: 2026-07-13 11:00
- 목적: 교수님 리스트의 `phi_liquidity`(미구현)를 마지막 이론적 후보로 검증. feature형·context형 둘 다 테스트하고, 레벨 특징의 방향성 이슈(과거 이진 Parkinson-magnitude 실패)가 연속에서 재현되는지 확인.
- 변경한 것: Amihud 비유동성 = log( mean_{20d}( |return| / trading_value ) ). baseline(5특징 orth) 위에 (a) feature(φ), (b) context(C)로 각각 추가.
- 고정 조건: κ=1, CPCV 45 split, split별 train-only 스케일링, 투자자별 HP, seed 42. numpy 재현.
- 데이터: `data/raw/samsung_macirl_EXTENDED_2019_2025.csv` (return, trading_value, vkospi). 삼성 자체 유동성(세 주체 공통).
- 결과 폴더: (별도 저장 없음, 스크립트 `/tmp/liq_model.py` 재현 가능).
- 주요 결과: **feature·context 둘 다 성능 악화 → 기각.** 특징/컨텍스트 탐색 정식 종결.

## 선별 (사전 상관)

| log-ILLIQ vs | 값 | 함의 |
| --- | ---: | --- |
| u_foreign | +0.009 | 방향 내용 ≈0 |
| u_institution | +0.039 | ≈0 |
| u_retail | −0.056 | 미약 |
| VKOSPI | +0.158 | 기존 기각된 위험정보와 중복 |
| \|momentum\| | −0.010 | 모멘텀과 무관 |

## 성능 (45 CPCV OOS, 방향정확도 / 강도상관)

| 구성 | 외국인 | 기관 | 개인 |
| --- | --- | --- | --- |
| base (5특징) | 0.628 / 0.259 | 0.520 / 0.040 | 0.609 / 0.248 |
| +illiq (feature) | 0.613 / 0.232 | 0.505 / 0.020 | 0.594 / 0.242 |
| +illiq (context) | 0.609 / 0.233 | 0.509 / 0.025 | 0.595 / 0.228 |

## 가중치 (illiq β, feature형)

| 대상 | illiq β | 부호일관성 |
| --- | ---: | ---: |
| 외국인 | +0.0042 | 71% |
| 기관 | +0.0007 | 73% |
| 개인 | −0.0164 | 93% |

- 해석:
  - **feature형 실패는 예견된 것.** log-ILLIQ와 부호 있는 u의 상관이 세 주체 ≈0(선별)이라 방향 판별력이 없음 → 과거 이진 Parkinson-magnitude 실패("레벨 특징이 매수/매도 방향과 정렬 안 됨")가 연속 강도 모델에서도 재현. 레벨 특징은 방향적 순flow 효과가 있을 때만 사는데, 유동성엔 그게 없음.
  - **context형 실패는 중복 때문.** ILLIQ가 VKOSPI와 0.158 겹침 → 이미 기각된 위험/변동성 정보를 재유입해 노이즈만 증가(외국인·개인 상관 하락).
- 결론: 유동성은 두 역할 모두에서 개선 없음 → 기각. **가진 데이터로의 특징/컨텍스트 탐색 정식 종결.** 최종 모델은 5특징(underwater·momentum·relative·herd(pooled)·shortmom_orth) 유지.
- 다음 액션: 무게중심을 검증 축으로 이동 — (1) Downstream 예측력(축 3, 최대 갭), (2) 타종목(SK하이닉스 등) 재현.
