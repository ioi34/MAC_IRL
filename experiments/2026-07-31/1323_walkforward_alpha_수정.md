# walk-forward가 α(컨텍스트 주효과)를 추정하지 않던 버그 수정

- 날짜/시간: 2026-07-31 13:23
- 상태: **코드 수정 완료 / 재실행 완료** (2026-07-31, `.venv`의 torch로 정식 실행)
- 목적: `2026-07-30/1629_bootstrap_alpha_수정.md`가 진단한 `context_main_effect`
  미전달 버그가 **부트스트랩에만 수정되고 walk-forward에는 적용되지 않았다.**
  그 결과 (1) α에는 V3 증거가 아예 없고, (2) V3의 β 값이 α 없는 축소 사양에서
  나온 값이다. 원고 §4.2의 V3 문장 전체가 여기에 걸린다.
- 변경한 것: **`scripts/train_continuous_walk_forward.py` 한 파일.** 모델·데이터·설정·
  하이퍼파라미터 일체 불변.
- 고정 조건: 3특징(momentum·persist·underwater), 맥락 2개, κ=1, λ=0.005 통일,
  walk-forward 3창(2023/2024/2025), seed 42, n=973.
- 데이터: `data/processed/dataset_continuous_reward3_persist.npz`
- 설정 파일: `configs/data_continuous.yaml`,
  `configs/features_continuous_reward3_persist.yaml`, `configs/model.yaml`,
  `configs/train.yaml`, `configs/experiment_continuous_reward3_persist_epochs75.yaml`
- 결과 폴더: `runs/continuous_reward3_persist_epochs75_validation/walk_forward_alphafix/`
  - 대조군(구, 덮어쓰지 않음): `runs/continuous_reward3_persist_epochs75_validation/walk_forward/`

## 근본 원인

`scripts/train_continuous_walk_forward.py:168`(수정 전)

```python
model = ContinuousInvestorIRLModel(
    num_features=len(feature_names),
    num_contexts=len(context_names),
    context_mask=context_mask,
)                       # context_main_effect 인자 없음 -> 기본값 False
```

`src/models/continuous.py:47`의 기본값이 `False`이므로, config가
`model.context_main_effect: true`를 지정해도 walk-forward는 **α 파라미터가 아예 없는
모델을 재적합**했다. `config_snapshot.yaml`에는 `context_main_effect: true`가
기록되어 있어 겉보기로는 정상이라 발견이 늦었다.

부트스트랩(`run_continuous_reward_validation.py:334`)과 CPCV 본학습
(`train_continuous.py:319`)은 정상. **모델 생성 지점 3곳을 전수 확인했고 남은
누락은 없다.**

| 생성 지점 | 경로 | α 전달 | 비고 |
| --- | --- | :---: | --- |
| CPCV 본학습 | `scripts/train_continuous.py:319` | ✅ | 애초에 정상 |
| 월별 블록 부트스트랩 | `scripts/run_continuous_reward_validation.py:334` | ✅ | 2026-07-30 수정 |
| walk-forward | `scripts/train_continuous_walk_forward.py:176` | ✅ | **본 수정** |
| (테스트) | `tests/test_continuous_behavior.py` | — | 의도된 인자 조합, 무관 |

ablation·ridge는 `_run_cpcv`가 `scripts.train_continuous`를 서브프로세스로 부르므로
정상 경로를 탄다.

## 수정 내용 (25 insertions, 1 deletion)

`train_continuous.py`의 처리를 그대로 옮겼다.

1. `_context_main_weights_frame`, `summarize_context_main_weights` import 추가
2. `context_main_effect`를 config에서 읽어 모델 생성자에 전달
3. 창별로 `context_main_frames` 수집
4. `context_main_weights.csv` / `context_main_weights_summary.csv` 저장

## 실행 (2026-07-31 완료)

`.venv`의 torch로 정식 실행. 3창 × 3유형 = 9 fit, 정상 종료.

```bash
PYTHONPATH=$PWD .venv/bin/python -m scripts.train_continuous_walk_forward \
  --data-config configs/data_continuous.yaml \
  --features-config configs/features_continuous_reward3_persist.yaml \
  --model-config configs/model.yaml \
  --train-config configs/train.yaml \
  --experiment-config configs/experiment_continuous_reward3_persist_epochs75.yaml \
  --output-dir runs/continuous_reward3_persist_epochs75_validation/walk_forward_alphafix \
  --test-years 2023 2024 2025
```

구 결과(`walk_forward/`)는 덮어쓰지 않았다 — 대조군으로 그대로 남아 있다.

### 확인 결과

1. **`walk_forward_alphafix/context_main_weights_summary.csv`가 생성됐다** — α의
   V3가 이번에 처음으로 존재한다.
2. **β의 V3 — 기관 persist의 부호 반전은 사라지지 않았다.** 연도별 원값이
   구·신 거의 그대로다(2023 −0.00128→−0.00096, 2024 +0.00702→+0.00593,
   2025 +0.00567→+0.00516) — α를 누락한 축소 사양의 산물이 아니라
   **모델 명세와 무관하게 실재하는 불안정성**이다.

```
   investor    feature  mean_old  mean_new  direction_consistency_old  direction_consistency_new  sign_reversal_old  sign_reversal_new
    foreign   momentum  0.054402  0.042437                   1.000000                   1.000000              False              False
    foreign    persist  0.053168  0.051645                   1.000000                   1.000000              False              False
    foreign underwater  0.006944  0.017506                   0.666667                   0.666667               True               True
institution   momentum -0.002110 -0.001537                   0.666667                   0.666667               True               True
institution    persist  0.003803  0.003375                   0.666667                   0.666667               True               True
institution underwater  0.010936  0.009131                   1.000000                   1.000000              False              False
     retail   momentum -0.052453 -0.049189                   1.000000                   1.000000              False              False
     retail    persist  0.043755  0.041013                   1.000000                   1.000000              False              False
     retail underwater  0.011272 -0.000221                   0.666667                   0.333333               True               True
```

(`sign_reversal`은 원본 스니펫에 없던 열이라 `positive_rate>0 and negative_rate>0`으로
직접 계산해 채웠다. `direction_consistency`는 세 창 중 최빈부호 비율.)

retail underwater는 연도별 부호 자체는 안 바뀌었다(2023 음, 2024·2025 양, 구·신
동일 패턴)이지만 크기가 이동해(2023 −0.0556→−0.0587, 2024 +0.0517→+0.0334,
2025 +0.0377→+0.0246) **평균의 부호가 미세하게 뒤집혔다**(+0.0113→−0.0002,
거의 0). `dominant_direction`은 평균 부호로 정하는 지표라 표기가 "positive→negative"로
바뀌었지만, 애초에 2/3 vs 1/3의 불안정한 계수였고 지금도 그렇다 — 새로운 불안정성이
아니라 원래도 약했던 계수가 0 근처에서 라벨만 넘어간 것이다.

## 가중치 결과

### α (V3, walk-forward 3창 평균·표준편차·방향일관성) — 이번에 처음 생성

| 대상 | 컨텍스트 | 평균 | 표준편차 | 방향일관성 | CPCV 45split α (참고) |
| --- | --- | ---: | ---: | ---: | --- |
| 외국인 | kospi_return_1d | +0.0456 | 0.0009 | 100% | +0.0331 (100%) |
| 외국인 | fx_level_z_252 | −0.0142 | 0.0391 | 66.7% | −0.0215 (97.8%) |
| 기관 | kospi_return_1d | −0.0148 | 0.0091 | 100% | −0.0060 (100%) |
| 기관 | fx_level_z_252 | −0.0049 | 0.0046 | 100% | −0.0035 (97.8%) |
| 개인 | kospi_return_1d | −0.0188 | 0.0167 | 66.7% | −0.0216 (100%) |
| 개인 | fx_level_z_252 | +0.0323 | 0.0206 | 100% | +0.0328 (100%) |

출처: `runs/continuous_reward3_persist_epochs75_validation/walk_forward_alphafix/context_main_weights_summary.csv`

→ **6개 전부 CPCV 45split α와 부호가 일치한다.** 외국인 fx와 개인 kospi만 3창 중
1창이 반대 부호라 방향일관성이 66.7%로 떨어진다 — n=3인 walk-forward는 45split
CPCV나 200회 부트스트랩보다 표본이 훨씬 작아 잡음에 민감한 게 당연하다. 부호
자체가 뒤집힌 계수는 없다.

### β (V3, 참고 — 위 대조표 원자료)

| 대상 | 피처 | 평균(신) | 방향일관성(신) | 부호반전(신) | 출처 |
| --- | --- | ---: | ---: | :---: | --- |
| 외국인 | momentum | +0.04244 | 100% | 아니오 | `walk_forward_alphafix/reward_weights_summary.csv` |
| 외국인 | persist | +0.05165 | 100% | 아니오 | 〃 |
| 외국인 | underwater | +0.01751 | 66.7% | 예 | 〃 |
| 기관 | momentum | −0.00154 | 66.7% | 예 | 〃 |
| 기관 | **persist** | **+0.00337** | **66.7%** | **예(유지)** | 〃 |
| 기관 | underwater | +0.00913 | 100% | 아니오 | 〃 |
| 개인 | momentum | −0.04919 | 100% | 아니오 | 〃 |
| 개인 | persist | +0.04101 | 100% | 아니오 | 〃 |
| 개인 | underwater | −0.00022 | 33.3%(평균 부호 기준) | 예(유지, 라벨만 이동) | 〃 |

## 해석

- **원고 §4.2의 핵심 논거(기관 persist의 walk-forward 불안정성)는 그대로 유효하다.**
  α 누락이 이 반전을 만들어낸 게 아니었다 — α를 포함한 정식 명세에서도 2023년
  창에서 부호가 뒤집힌다. 축소 사양의 인공물이라는 가설은 기각됐다.
- **α는 이번에 처음으로 walk-forward 근거를 얻었고, CPCV 점추정과 전부 부호가
  일치한다.** n=3의 한계로 일관성 수치 자체는 낮게 나올 수 있지만(외국인 fx,
  개인 kospi), 이는 표본 크기의 문제이지 부호 불일치가 아니다.
- **retail underwater의 평균 부호 반전은 별개 현상으로 분리해서 봐야 한다.** 기관
  persist처럼 "새로 드러난 불안정성"이 아니라, 원래도 2/3 대 1/3으로 약했던
  계수가 크기 재조정으로 평균 부호 라벨만 넘어간 것이다. 원고에서 이 계수를
  강한 근거로 쓰고 있지 않다면 추가 조치는 불필요하다.

## 다음 액션

1. ~~위 명령으로 재실행~~ → **완료.**
2. ~~β의 V3 반전 여부가 바뀌면 동료에게 즉시 공유~~ → **반전 유지, 원고 §4.2 근거
   그대로 사용 가능.** 별도 공유 불필요.
3. `scripts/apply_1col_layout.py`의 `ALPHA_ROWS`에서 V3 열 `---`를 위 α 표의
   실제 값으로 교체 (아직 미착수 — 다음 세션에서).
4. 원고 §4.2 / Table 2·3의 V3 수치를 위 표로 갱신 (아직 미착수).
5. retail underwater 평균 부호 반전이 원고 어디에 인용되는지 확인 — 인용
   없으면 조치 불필요, 있으면 "라벨 이동, 원래도 약함"으로 각주 필요.
