# 전체 재실험 결과 요약

- 날짜/시간: 2026-06-26 17:24 KST
- 목적: 과거에 진행한 주요 실험을 다시 실행하고, 추가 ablation으로 어떤 변경이 성능에 영향을 주는지 비교한다.
- 변경한 것: 손실 feature, herd 정의, persist 추가, volatility 제거, 데이터 기간을 각각 바꿔 비교했다.
- 고정 조건: MAC-IRL fixed linear reward, 투자자별 독립 모델, CPCV 10 folds / test folds 2 / purge 1 / embargo 5, seed 42, 200 epochs, batch size 256, learning rate 0.01, L1 0.001.
- 데이터: 삼성전자 수급 데이터. 2021-2025 실험은 973개 표본, 2019-2025 실험은 1467개 표본.
- 설정 파일: `experiments/2026-06-26/configs/`
- 결과 폴더: `runs/exp_20260626_*`, 과거 loss-aversion 재실행 결과는 `experiments/2026-06-26/loss_aversion_67fe2d6/`

## 전체 결과

| 실험 | 변경한 것 | 표본 | Accuracy | Macro F1 | NLL | 결과 폴더 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 손실회피 이평선 | 과거 코드 `loss_aversion + herd + momentum + volatility` | 973 | 0.5682 | 0.5368 | 0.6804 | `experiments/2026-06-26/loss_aversion_67fe2d6/` |
| 2021 underwater single herd | 평단가 기반 underwater, 단일 herd | 973 | 0.5760 | 0.5544 | 0.6790 | `runs/exp_20260626_single_herd_2021_current_label/` |
| 2021 underwater single herd, volatility 제거 | 위 조건에서 volatility 제거 | 973 | 0.5868 | 0.5722 | 0.6746 | `runs/exp_20260626_single_herd_no_volatility_2021_current_label/` |
| 2021 single herd + persist | 단일 herd 유지, persist 추가 | 973 | 0.5762 | 0.5568 | 0.6796 | `runs/exp_20260626_single_herd_persist_2021_current_label/` |
| 2021 pairwise herd, persist 없음 | herd를 투자자쌍별로 분리, persist 없음 | 973 | 0.5766 | 0.5578 | 0.6845 | `runs/exp_20260626_pairwise_no_persist_2021_current_label/` |
| 2021 pairwise herd + persist | 투자자쌍별 herd와 persist 함께 사용 | 973 | 0.5760 | 0.5576 | 0.6851 | `runs/exp_20260626_pairwise_persist_2021_current_label/` |
| 2019 underwater single herd | 2019-2025 확장 데이터, 단일 herd | 1467 | 0.5693 | 0.5600 | 0.6793 | `runs/exp_20260626_single_herd_2019_current_label/` |
| 2019 pairwise herd + persist | 2019-2025 확장 데이터, pairwise herd + persist | 1467 | 0.5633 | 0.5547 | 0.6848 | `runs/exp_20260626_pairwise_persist_2019_current_label/` |

## 해석

가장 좋은 결과는 `2021 underwater single herd, volatility 제거` 실험이다.

- Accuracy: 0.5868
- Macro F1: 0.5722
- NLL: 0.6746

즉 현재 기준에서는 `volatility`가 성능을 돕기보다 방해하는 쪽에 가깝다. Notion에서 volatility를 제거 후보로 적어둔 판단이 이번 재실험에서도 맞게 나왔다.

`underwater`는 과거 `loss_aversion`보다 성능이 좋다. 다만 이번 loss-aversion 재실행은 과거 commit `67fe2d6` 기준이고, 현재 underwater 실험은 라벨링 방식 변경 이후 코드이므로 완전한 단일 조건 비교는 아니다. 그래도 지금까지의 기록과 재실험 결과를 함께 보면 메인 손실 feature는 `underwater` 유지가 더 타당하다.

`persist`는 단일 herd 모델에 붙였을 때 Macro F1을 0.5544에서 0.5568로 조금 올렸지만, Accuracy와 NLL 개선은 거의 없다.

`pairwise herd`는 Macro F1을 약간 올리지만 NLL을 나쁘게 만든다. 특히 `pairwise herd + persist`는 해석 feature는 늘어나지만 예측 손실 기준으로는 단일 herd보다 불리하다.

2019-2025 확장 데이터는 표본이 973개에서 1467개로 늘었지만, Accuracy는 오히려 낮아졌다. 다만 Macro F1은 2021 single herd보다 조금 높다. 장기 데이터가 안정성을 주는지 보려면 기간별 regime 분석이 필요하다.

## 다음 액션

1. 메인 후보는 `underwater + herd + momentum`, 즉 volatility 제거 모델로 둔다.
2. 논문 표에는 `loss_aversion`, `underwater single herd`, `underwater no volatility`, `pairwise herd + persist`를 핵심 비교로 넣는 것이 좋다.
3. 다음 실험은 2019-2025 데이터에서도 volatility 제거를 실행해 데이터 기간 확장과 volatility 제거 효과를 분리한다.
4. 기관 성능은 여전히 약하므로 기관 전용 feature나 라벨링 기준을 따로 검토한다.

## 근거 구분

- 실제 파일 기반: `runs/exp_20260626_*`, `experiments/2026-06-26/loss_aversion_67fe2d6/`
- 과거 코드 기반: loss-aversion 재실행은 git commit `67fe2d6`을 별도 worktree에서 실행했다.
- Notion/GitHub 기록 참고: 과거 실험 목록과 원래 의도 확인에만 사용했다.
