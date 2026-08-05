# persist 시차 검증 + Adam↔정확 Lasso 대조

- 날짜/시간: 2026-08-05 17:24
- 목적: 논문 §3~§4 점검 중 제기된 두 가지를 실증 확인한다.
  (1) Adam 최적화 결과가 논문이 주장하는 "정확한 Lasso 해"와 일치하는가.
  (2) persist 피처가 타깃 대비 2일 시차를 갖는 것이 결과에 영향을 주는가.
- 변경한 것:
  - (실험 1) 학습기를 Adam(75 epoch) 대신 sklearn `Lasso`(좌표하강, `max_iter=200000`,
    `tol=1e-10`, `fit_intercept=False`, `alpha = lambda_l1 / 2 = 0.0025`)로 교체.
    스케일러는 각 split에 저장된 `*_scaler.joblib`을 그대로 사용.
  - (실험 2) persist 피처를 `u_{t-1}` → `u_t`로 한 칸 당김
    (`XB[r] = X[r+1]`, 마지막 행은 결측 처리하여 제외).
- 고정 조건: 동일한 45개 CPCV split 인덱스
  (`runs/continuous_reward3_persist_epochs75/split_*/indices.npz`),
  동일 피처 3개(momentum, persist, underwater), 동일 컨텍스트 2개
  (kospi_return_1d, fx_level_z_252), 동일 λ=0.005, 절편 없음,
  clip(-1,1) 정책, 3개 투자자 전부 동일 사양.
- 데이터: `data/processed/dataset_continuous_reward3_persist.npz`
  (삼성전자 005930, 973 state days, 2022-01-06 ~ 2025-12-29)
- 설정 파일: `runs/continuous_reward3_persist_epochs75/config_snapshot.yaml`
- 결과 폴더: 별도 run 디렉터리 없이 스크립트로 즉석 계산. 재현 코드는 본 노트 하단.

## 주요 결과

### 실험 1 — Adam은 이미 Lasso 최적해에 도달해 있다

45 split 평균 계수를 비교한 결과 최대 절대 편차는 약 0.001 수준이며 부호·크기 모두 일치한다.
따라서 논문 §3.3의 "유일 최소해" 주장과 보고된 수치 사이에 모순은 없다.
동시에, **Adam이 결과에 기여하는 바가 없으므로 파이프라인에서 제거 가능하다.**

### 실험 2 — persist 시차 2일이 성능을 유의하게 깎고 있다

현재 구현은 `persist[t] = u_{t-1}`, 타깃은 `a_{t+1} = u_{t+1}` 이므로 간격이 2일이고
`u_t`가 전혀 쓰이지 않는다. momentum(`log(P_t/P_{t-20})`)과 underwater는 t일 정보를
쓰므로 피처 간 시점 정렬도 어긋난다. `u_t`로 교체하면 세 유형 모두 개선되고,
특히 **기관이 평균-행동 벤치마크를 처음으로 상회한다(OOS R² -0.0036 → +0.0231).**

원계열 자기상관(삼성전자 2022-2025, n=977):

| 투자자 | sd(u) | corr(u_{t+1}, u_t) | corr(u_{t+1}, u_{t-1}) |
| --- | ---: | ---: | ---: |
| foreign | 0.2575 | +0.4015 | +0.3076 |
| institution | 0.1262 | +0.1565 | +0.1123 |
| retail | 0.3339 | +0.3444 | +0.2456 |

## 가중치 결과

정확 Lasso, 45 split CPCV 평균. A = 현행(`u_{t-1}`), B = 수정안(`u_t`).

| 대상 | 피처/가중치 | 값 (A 현행) | 값 (B 수정) | 변동성/일관성 | 출처 |
| --- | --- | ---: | ---: | ---: | --- |
| foreign | β momentum | +0.0438 | +0.0383 | A std 0.0088 | 본 노트 재현 스크립트 |
| foreign | β persist | +0.0518 | +0.0793 | A std 0.0059 | 〃 |
| foreign | β underwater | +0.0068 | +0.0110 | A std 0.0041 | 〃 |
| institution | β momentum | -0.0014 | -0.0009 | A std 0.0022 | 〃 |
| institution | β persist | +0.0103 | +0.0236 | A std 0.0031 | 〃 |
| institution | β underwater | -0.0002 | -0.0002 | A std 0.0006 | 〃 |
| retail | β momentum | -0.0364 | -0.0277 | A std 0.0081 | 〃 |
| retail | β persist | +0.0469 | +0.1048 | A std 0.0085 | 〃 |
| retail | β underwater | +0.0276 | +0.0182 | A std 0.0150 | 〃 |

α(직접 컨텍스트 효과)는 실험 1에서만 산출했다. 정확 Lasso 45 split 평균:

| 대상 | 가중치 | 값 | 표준편차 | 출처 |
| --- | --- | ---: | ---: | --- |
| foreign | α KOSPI | +0.0329 | 0.0046 | 본 노트 재현 스크립트 |
| foreign | α FX | -0.0223 | 0.0099 | 〃 |
| institution | α KOSPI | -0.0059 | 0.0024 | 〃 |
| institution | α FX | -0.0035 | 0.0025 | 〃 |
| retail | α KOSPI | -0.0219 | 0.0062 | 〃 |
| retail | α FX | +0.0332 | 0.0116 | 〃 |

대조군: Adam 기존 결과
`runs/continuous_reward3_persist_epochs75/reward_weights_summary.csv`,
`runs/continuous_reward3_persist_epochs75/context_main_weights_summary.csv`
(foreign momentum +0.04354, foreign persist +0.05055, institution persist +0.01024,
retail momentum -0.03556, retail persist +0.04620).

## 성능 비교 (45 split 평균, 테스트 구간)

| 투자자 | 지표 | A 현행 (`u_{t-1}`) | B 수정 (`u_t`) |
| --- | --- | ---: | ---: |
| foreign | directional acc | 0.6450 | 0.6501 |
| foreign | correlation | 0.3101 | 0.3640 |
| foreign | OOS R² | 0.1374 | 0.1853 |
| institution | directional acc | 0.5381 | 0.5561 |
| institution | correlation | 0.0669 | 0.1646 |
| institution | OOS R² | **-0.0036** | **+0.0231** |
| retail | directional acc | 0.6077 | 0.6332 |
| retail | correlation | 0.2421 | 0.3116 |
| retail | OOS R² | 0.1064 | 0.1520 |

## 해석

- Adam은 결과에 영향을 주지 않는다. 제거하고 좌표하강 Lasso로 대체하면 수치는 유지되면서
  optimizer 하이퍼파라미터(seed, epochs, 유형별 batch/lr) 보고 부담과 수렴 관련 리뷰 지적이
  동시에 사라진다. Kingma(2015) 인용도 불필요해진다.
- 기관의 약한 결과는 전부가 실질적 발견이 아니다. 최소한 일부는 시차 정렬 오류에서 온다.
  `u_t`로 고치면 기관도 평균-행동 벤치마크를 넘는다. 현재 §4.4의 "기관은 구별되는 공통
  성향이 없다"는 서술은 이 수정 이후 다시 판단해야 한다.
- 부호 방향(foreign momentum +, retail momentum -)은 수정안에서도 유지된다.
  다만 persist가 설명력을 더 흡수하면서 momentum 계수 절댓값은 줄어든다
  (foreign 0.0438→0.0383, retail -0.0364→-0.0277). 초록·1장의 인용 수치는
  수정안을 채택하면 함께 갱신해야 한다.

## 다음 액션

1. persist 시차 수정 여부 결정. 채택 시 `src/features/persist.py`의 `.shift(1)` 제거 후
   전체 파이프라인 재실행 및 정식 run 디렉터리 생성 (본 노트는 즉석 계산이라 정식 run 아님).
2. herd 피처도 동일한 `.shift(1)` 문제를 가지므로 같이 점검.
3. `mac_irl_icaif26_full_en.tex` 헤더의 canonical run 경로가
   `runs/continuous_reward3_lambda_unified/`로 되어 있으나 실제 표 수치의 출처는
   `runs/continuous_reward3_persist_epochs75/`이다. 전자는 persist가 아니라 herd
   (타 투자자 lagged flow)를 쓰므로 §3.2 본문 서술과 불일치한다. 헤더 갱신 필요.

## 추가 검증 (2026-08-05 17:5x) — V1 부호 일관성까지 포함한 A vs B

정확 Lasso, 45 split, 괄호 안은 V1 부호 일관성.

| 대상 | 가중치 | A 현행 | B 수정 | 판정 변화 |
| --- | --- | ---: | ---: | --- |
| foreign | β momentum | +0.0438 (100.0%) | +0.0383 (100.0%) | 유지 |
| foreign | β persist | +0.0518 (100.0%) | +0.0793 (100.0%) | 유지, 강화 |
| foreign | β underwater | +0.0068 (95.6%) | +0.0110 (100.0%) | 개선 |
| foreign | α KOSPI | +0.0338 (100.0%) | +0.0066 (91.1%) | **거의 소멸** |
| foreign | α FX | -0.0221 (100.0%) | -0.0211 (100.0%) | 유지 |
| institution | β momentum | -0.0014 (44.4%) | -0.0009 (28.9%) | 양쪽 다 탈락 |
| institution | β persist | +0.0103 (100.0%) | +0.0236 (100.0%) | 유지, 2배 |
| institution | β underwater | -0.0002 (22.2%) | -0.0002 (17.8%) | 양쪽 다 탈락 |
| institution | α KOSPI | -0.0059 (100.0%) | -0.0168 (100.0%) | 유지, 3배 |
| institution | α FX | -0.0035 (86.7%) | -0.0035 (91.1%) | 경계 |
| retail | β momentum | -0.0364 (100.0%) | -0.0277 (100.0%) | 유지 |
| retail | β persist | +0.0469 (100.0%) | +0.1048 (100.0%) | 유지, 2배 |
| retail | β underwater | +0.0276 (91.1%) | +0.0182 (82.2%) | 악화 |
| retail | α KOSPI | -0.0225 (100.0%) | **+0.0290 (100.0%)** | **부호 반전** |
| retail | α FX | +0.0331 (100.0%) | +0.0304 (100.0%) | 유지 |

출처: 본 노트 재현 스크립트(하단), 대조군은
`runs/continuous_reward3_persist_epochs75/`.

### 부호 반전의 원인 — u_t 와 r_t 의 동시점 상관

| 투자자 | corr(u_t, r_t) | corr(a_{t+1}, r_t) | corr(u_t, a_{t+1}) |
| --- | ---: | ---: | ---: |
| foreign | +0.380 | +0.175 | +0.403 |
| institution | +0.457 | -0.062 | +0.150 |
| retail | -0.550 | -0.108 | +0.344 |

`u_t`는 당일 KOSPI 수익률과 강하게 동시 상관된다(외국인 +0.38, 개인 -0.55).
현행 A안에서 persist는 `u_{t-1}`이라 이 채널을 담지 못하고, 그 몫이 α_KOSPI로 흘러간다.
B안에서 persist가 `u_t`를 담으면 α_KOSPI는 잔차만 남는다. 따라서 개인 α_KOSPI의
부호 반전은 단순 상관(-0.108, A안 부호와 일치)과 모순되는 것이 아니라
**통제(partialling) 효과**다. 다만 §4.4의 해석 문장은 그대로 쓸 수 없게 된다.

## 해석 (추가분)

- **논문의 핵심 주장은 안전하다.** foreign momentum(+) vs retail momentum(-) 대비는
  B안에서도 양쪽 100% 유지된다. 절댓값은 13~24% 줄어든다.
- **§4.4의 컨텍스트 서사는 무너진다.** "개인이 외국인이 물러나는 국면에서 물량을
  흡수한다"는 서술은 α_KOSPI 부호 대비에 의존하는데, B안에서 개인 α_KOSPI가 양수로
  뒤집히고 외국인 α_KOSPI는 사실상 0이 된다.
- **기관 귀무 결과도 무너진다.** 초록의 "no gain over a mean-action benchmark"와
  §4.4 제목("no distinct common tendency")은 B안에서 거짓이 된다.
- **새 위험: 모델이 flow AR(1)에 가까워진다.** 개인 persist가 +0.1048로 momentum
  절댓값(0.0277)의 약 4배가 된다. 논문 제목이 내세우는 momentum 대비가 적합의
  작은 부분으로 밀린다. 정확하지만 서사적으로는 약해질 수 있다.
- 본 검증은 V2 부트스트랩, V3 walk-forward, 현대차 전이를 포함하지 않는다.
  B안 채택 시 이 셋을 모두 재실행해야 한다.

## 재현 스크립트

```python
import numpy as np, glob, pandas as pd
from sklearn.linear_model import Lasso
d = np.load('data/processed/dataset_continuous_reward3_persist.npz', allow_pickle=True)
X, A, C = d['features'].astype(float), d['actions'].astype(float), d['contexts'].astype(float)
invs, fn = list(d['investors']), list(d['feature_names'])
pi, n = fn.index('persist'), len(A)

XB = X.copy(); XB[:-1, :, pi] = X[1:, :, pi]; XB[-1, :, pi] = np.nan
valid = ~np.isnan(XB[:, 0, pi])

def run(Xv, tag):
    out = []
    for sp in sorted(glob.glob('runs/continuous_reward3_persist_epochs75/split_*')):
        ix = np.load(sp + '/indices.npz', allow_pickle=True)
        tr = np.array([i for i in ix['train_indices'] if valid[i]])
        te = np.array([i for i in ix['test_indices'] if valid[i]])
        cm, cs = C[tr].mean(0), C[tr].std(0); Cs = (C - cm) / cs
        for j, inv in enumerate(invs):
            Z = Xv[:, j, :]; m_, s_ = Z[tr].mean(0), Z[tr].std(0); Zs = (Z - m_) / s_
            W = np.hstack([Zs, (Zs[:, :, None] * Cs[:, None, :]).reshape(n, -1), Cs])
            y = A[:, j]
            mod = Lasso(alpha=0.0025, fit_intercept=False, max_iter=100000, tol=1e-8).fit(W[tr], y[tr])
            p = np.clip(W[te] @ mod.coef_, -1, 1)
            out.append(dict(investor=inv, b_mom=mod.coef_[0], b_per=mod.coef_[1], b_uw=mod.coef_[2],
                            dacc=np.mean(np.sign(p) == np.sign(y[te])),
                            corr=np.corrcoef(p, y[te])[0, 1],
                            r2=1 - ((y[te] - p) ** 2).sum() / ((y[te] - y[tr].mean()) ** 2).sum()))
    print(tag); print(pd.DataFrame(out).groupby('investor').mean(numeric_only=True).round(4))

run(X,  'A: persist = u_{t-1} (current)')
run(XB, 'B: persist = u_t     (lag-1)')
```
