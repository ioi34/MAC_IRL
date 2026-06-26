# 4단계 액션 라벨링 초기 실험

- 날짜/시간: 2026-06-26 17:39 KST
- 목적: 기존 2-class `sell/buy` 라벨을 4-class `strong_sell/weak_sell/weak_buy/strong_buy`로 바꾼 뒤, 데이터 생성과 학습이 정상 동작하는지 확인한다.
- 변경한 것: 라벨링을 rolling median 기준 2분류에서 rolling quartile 기준 4분류로 변경했다.
- 고정 조건: 2021-2025 삼성전자 데이터, 973개 표본, CPCV 45 splits, 200 epochs, L1 0.001.
- 데이터: `samsung_macirl_EXTENDED_2021_2025.csv`
- 설정 파일: `configs/data_extended.yaml`, `configs/features_underwater.yaml`, `experiments/2026-06-26/configs/experiment_four_class_pairwise_persist_2021.yaml`
- 결과 폴더: `runs/exp_20260626_four_class_pairwise_persist_2021/`

## 라벨 정의

과거 252거래일 투자자별 flow 분포의 25%, 50%, 75% 분위수를 사용했다.

| 조건 | 라벨 | 인덱스 |
| --- | --- | ---: |
| flow < q25 | strong_sell = -2 | 0 |
| q25 <= flow < q50 | weak_sell = -1 | 1 |
| q50 <= flow < q75 | weak_buy = 1 | 2 |
| q75 <= flow | strong_buy = 2 | 3 |

## 라벨 분포

| 투자자 | strong sell | weak sell | weak buy | strong buy |
| --- | ---: | ---: | ---: | ---: |
| foreign | 246 | 229 | 234 | 264 |
| institution | 227 | 243 | 241 | 262 |
| retail | 263 | 241 | 228 | 241 |

네 클래스가 대체로 균형 있게 나뉘었다.

## 결과

| 투자자 | Accuracy | Macro F1 | NLL |
| --- | ---: | ---: | ---: |
| foreign | 0.3557 | 0.2247 | 1.3483 |
| institution | 0.2883 | 0.1861 | 1.3925 |
| retail | 0.3214 | 0.2025 | 1.3757 |
| 평균 | 0.3218 | 0.2044 | 1.3722 |

4-class 랜덤 기준은 Accuracy 약 0.25, NLL 약 1.386이다. 평균 Accuracy와 NLL은 랜덤보다 조금 낫지만, Macro F1은 낮다. 특히 institution은 NLL이 1.3925로 랜덤 기준보다 나쁘다.

## 해석

4-class 라벨링과 학습 파이프라인은 정상 동작한다. 다만 기존 2-class보다 문제 난도가 올라가서 현재 feature 조합으로는 weak/strong 강도를 충분히 분리하지 못한다.

가중치 방향은 일부 안정적이다.

- foreign: momentum 양수 100%, underwater 음수 93.3%
- institution: underwater 음수 97.8%, momentum 음수 97.8%, volatility 양수 100%, persist 양수 95.6%
- retail: momentum 음수 100%, persist 양수 100%, herd_from_foreign/institution 양수 93.3%

## 다음 액션

1. 4-class에서도 이전에 가장 좋았던 `volatility 제거` 조합을 다시 실험한다.
2. strong/weak 분리를 위해 action value를 `[-1.5, -0.5, 0.5, 1.5]`처럼 완화하는 실험을 고려한다.
3. 4-class 결과는 Accuracy만 보지 말고 Macro F1과 confusion matrix를 함께 확인한다.
4. institution은 4-class에서 특히 약하므로 별도 feature 또는 라벨 기준을 검토한다.
