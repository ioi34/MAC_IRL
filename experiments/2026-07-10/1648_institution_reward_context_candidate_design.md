# 기관 연속행동 보상특징·컨텍스트 후보 설계

- 날짜/시간: 2026-07-10 16:48 KST
- 실행 상태: **미실행** — 기존 연속 모델을 새 평가기준으로 진단하고 후속 실험을 설계했으며 새 학습은 수행하지 않았다.
- 목적: 실제 기관 순매수 강도를 설명하면서 계수의 경제적 의미를 논문에서 직접 해석할 수 있는 보상특징과 컨텍스트 후보를 선정한다.
- 변경한 것: 코드·데이터·학습 결과 변경 없음. 기존 OOS 예측을 이용해 기관 기준선을 NRMSE, 평균 대비 skill, 방향 Balanced Accuracy·Macro F1, 상관, 분산 재현, 꼬리·연도 안정성으로 추가 진단했다.
- 고정 조건: 973표본(2022-01-06~2025-12-29), CPCV 45 split, seed 42, 기존 연속 타깃 `(buy-sell)/(buy+sell)`.
- 데이터: `data/processed/dataset_continuous_final.npz`, `samsung_macirl_EXTENDED_2019_2025.csv`
- 설정 파일: `runs/continuous_net_buy_behavior/config_snapshot.yaml`
- 결과 폴더: `runs/continuous_net_buy_behavior`
- 주요 결과: 기관 RMSE는 낮아 보이지만 실제 표준편차도 작아서 NRMSE가 1.006이고 학습구간 평균 대비 RMSE 개선은 0.51%뿐이다. 방향 Balanced Accuracy 0.517, Pearson 0.077, 예측/실제 표준편차 비율 0.069로 현재 특징은 기관 행동을 거의 설명하지 못한다.

## 현재 기관 기준선에 평가기준 적용

| 기준 | 현재 값 | 판단 |
| --- | ---: | --- |
| RMSE | 0.124611 | 투자자 간 직접 비교 금지 |
| NRMSE(test 표준편차 기준) | 1.006056 | 단순 평균 수준 |
| 학습 평균 대비 RMSE skill | 0.005060 | 0.51% 개선에 불과 |
| 방향 Accuracy | 0.534066 | 약함 |
| 방향 Balanced Accuracy | 0.517181 | 무작위 수준에 근접 |
| 방향 Macro F1 | 0.506616 | 약함 |
| Pearson / Spearman | 0.077149 / 0.078178 | 강도 순위 설명력도 약함 |
| 예측 표준편차/실제 표준편차 | 0.069445 | 심각한 과소분산 |
| 절대행동 상위 25% 방향정확도 | 0.584362 | 큰 행동에서만 제한적 신호 |

연도별 방향정확도는 2022년 0.593, 2023년 0.510, 2024년 0.533, 2025년 0.539다. seed는 42 하나뿐이므로 seed 안정성은 아직 평가할 수 없다.

## 이론 기반 보상특징 후보

| 우선순위 | 후보 | 정의 예시 | 사전 가설/예상 부호 | 구현 |
| --- | --- | --- | --- | --- |
| P1 | `own_flow_persistence_3` | 기관 행동의 3일 EWMA, 시점 t까지만 사용 | β>0: 주문 분할·집행 지속 | 현재 데이터 |
| P1 | `short_residual_return_1/5` | 삼성 1·5일 수익률 − KOSPI200 동기간 수익률 | β<0: 국내 기관의 일별 contrarian·리밸런싱 | 현재 데이터 |
| P1 | `benchmark_drift_20` | 삼성 20일 누적수익률 − KOSPI200 누적수익률 | β<0: 벤치마크 대비 비중 drift 복원 | 현재 데이터 |
| P1 | `overnight_global_signal` | 전일 S&P500·SOX 수익률 또는 예상 밖 글로벌 뉴스 | β>0: 글로벌 정보에 대한 momentum 반응 | 외부 데이터 필요 |
| P2 | `foreign_flow_lag1`, `retail_flow_lag1` | 외국인·개인 전일 순매수 불균형을 분리 | 외국인 β<0: 국내 기관의 반대편 거래; 개인 부호는 개방 가설 | 현재 데이터 |
| P2 | `domestic_fund_flow_pressure` | 국내 주식형 펀드·ETF 순유입/AUM | β>0: 유입 시 강제 매수, 환매 시 강제 매도 | 외부 데이터 필요 |
| P2 | `public_information_signal` | 애널리스트 이익추정치 변화·실적 surprise | β>0: 공개정보 해석에 따른 방향 거래 | 외부 데이터 필요 |
| P3 | `index_rebalance_demand` | KOSPI200 목표비중 변화 × 추종 AUM | β>0: 벤치마크 수요 방향 추종 | 외부 데이터 필요 |

- `persist`는 기존 바이너리 기관 실험에서 β=0.0067, 양(+) 100%로 가장 직접적인 로컬 선행근거가 있다.
- 기존 `relative(20)` 단독 추가는 바이너리 기관 Accuracy를 0.0128 낮췄다. 따라서 `benchmark_drift`는 단독 채택하지 않고 기간말 컨텍스트와 결합한 사전가설로 재검증한다.
- 거래대금 20·60·120일을 동시에 넣은 과거 실험은 기관 Accuracy를 0.0311 낮췄다. 거래대금 자체를 방향 보상으로 다시 넣지 않고 유동성 컨텍스트로만 사용한다.

## 컨텍스트 후보와 허용할 상호작용

현재 모델의 컨텍스트는 직접 절편이 아니라 `B(feature, context)`로 보상계수를 바꾼다. 모든 조합을 자동 생성하지 않고 아래 이론 대응만 허용하는 masked B가 해석과 과적합 방지에 적합하다.

| 우선순위 | 컨텍스트 | 연결할 보상특징 | 예상 상호작용 |
| --- | --- | --- | --- |
| P1 | `vkospi_return_1d`, `vkospi_level_z` | persistence, short residual return | 위험·유동성 충격이 주문 지속성과 contrarian 강도를 조절 |
| P1 | `liquidity_capacity_20` | persistence, public information signal | 유동성이 높을수록 신호 집행 강도 증가; Amihud 부호를 뒤집어 양수가 고유동성이 되게 정의 |
| P1 | `quarter_end_intensity` | benchmark drift | B<0: 기간말에 벤치마크 비중 복원 강화 |
| P2 | `q4_reporting_window` | medium momentum | B>0: 연말 winner 매수·loser 매도의 window dressing 가설 |
| P2 | `market_downside_regime` | persistence, short residual return | 하락장에서 risk-budget 축소 또는 contrarian 강화라는 경쟁 가설 |
| P2 | `earnings_window` | public information signal | 실적 발표 전후 정보 신호 민감도 확대 |
| P3 | `fund_flow_stress` | persistence, benchmark drift | 환매 압력에서 집행 지속·리밸런싱 반응 변화 |

VKOSPI 변화는 과거 바이너리 결과에서 KOSPI 컨텍스트와 함께 사용할 때 기관 Accuracy가 0.5281에서 0.5312로 상승했고, 전체 컨텍스트 조합에서는 0.5385였다. 연속 버전에서는 단독 ablation으로 다시 확인해야 한다.

## 가중치 결과

새 후보는 미실행이므로 학습 가중치가 없다. 아래는 비교 기준인 현재 기관 연속 모델의 실제 값이다.

| 대상 | 피처/가중치 | 값 | 변동성/일관성 | 출처 |
| --- | --- | ---: | ---: | --- |
| institution | beta underwater | -0.001049 | std 0.001016 / 음 91.1% | `runs/continuous_net_buy_behavior/reward_weights_summary.csv` |
| institution | beta herd | -0.004210 | std 0.000589 / 음 100.0% | 위와 같음 |
| institution | beta momentum | -0.000724 | std 0.001265 / 음 73.3% | 위와 같음 |
| institution | beta volatility | 0.002645 | std 0.001865 / 양 91.1% | 위와 같음 |

## 실험·채택 기준

1. 각 후보는 기존 4개 특징에 하나씩만 추가하고 동일 outer split에서 paired 비교한다.
2. 1차 채택: 평균 RMSE skill 개선, outer split 승률 60% 이상, 계수 부호 일관성 80% 이상을 모두 만족한다.
3. 방향 Balanced Accuracy·Macro F1, Pearson·Spearman, 예측/실제 표준편차 비율, 상위 25% 행동 방향정확도를 함께 보고 한 지표만 좋아진 후보는 채택하지 않는다.
4. 2022~2025 연도별 walk-forward와 최소 5개 seed에서 부호·성능 방향을 확인한다.
5. P1 통과 후보만 조합하고 nested tuning은 조합 후 학습구간 안에서 수행한다.
6. 컨텍스트는 사전 지정된 feature-context 쌍만 허용하며, 여러 후보 탐색 시 BH-FDR과 계수 부호 안정성을 함께 보고한다.

- 해석: 현 `institution`은 금융투자·보험·투신·은행·연기금 등의 혼합일 수 있어 서로 다른 전략이 상쇄될 가능성이 크다. 가능하면 기관 세부주체별 순매수로 분리하는 것이 특징 추가보다 우선이다.
- 다음 액션: P1 순서를 `persistence → short residual return → benchmark drift×quarter-end → VKOSPI → liquidity×persistence`로 고정해 단일 ablation을 실행한다. 외부 데이터 단계에서는 글로벌 신호와 펀드 플로우를 우선 수집한다.

## 문헌 근거

- 국내 기관 세부유형의 일별 herding·contrarian: [KCI 기관 투자자 유형별 거래행태 분석](https://www.kci.go.kr/kciportal/ci/sereArticleSearch/ciSereArtiView.kci?sereArticleSearchBean.artiId=ART002036865)
- 기관 herding·feedback trading: [Nofsinger and Sias (1999)](https://doi.org/10.1111/0022-1082.00188)
- 기관 주문 집행비용·거래스타일: [Keim and Madhavan (1997)](https://doi.org/10.1016/S0304-405X(97)00031-7)
- 펀드 flow에 의한 강제매매: [Coval and Stafford](https://www.nber.org/papers/w11357)
- 기관의 정보거래·유동성 공급: [Da, Gao, and Jagannathan](https://www.nber.org/papers/w14609)
- 연말 window dressing: [Lakonishok et al.](https://www.nber.org/papers/w3617)
- 한국 기관의 거시뉴스 반응: [The response of different investor types to macroeconomic news](https://doi.org/10.1016/j.mulfin.2019.02.005)

## 근거 구분

- 실제 파일 기반: 현재 기관 평가값, 기존 가중치, 과거 VKOSPI·relative·turnover 실험 결과.
- 논문 근거: 후보별 기관행동 가설과 예상 부호.
- 코드·설정에서 추론한 항목: 현재 컨텍스트가 직접효과가 아니라 feature slope를 바꾸므로 masked B가 필요하다는 점.
- 미실행 항목: 모든 신규 보상특징·컨텍스트의 학습 성능과 가중치.
