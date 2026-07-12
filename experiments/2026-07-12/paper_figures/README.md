# 논문·GitHub용 실험 그래프 모음

`runs/`가 `.gitignore`에 포함되어 있어 기존 실행 폴더의 그래프는 GitHub에 올라가지 않는다. 이 폴더는 현재 연구에서 사용한 그림과 그래프 원자료를 `experiments/` 아래에 모은 Git 추적용 사본이다. 원본 실행 결과는 변경하지 않았다.

## 폴더 구성

| 번호 | 폴더 | 내용 | 논문 권장 위치 |
| --- | --- | --- | --- |
| 01 | `01_continuous_all_investors/` | 외국인·기관·개인의 연속 순매수 실제값과 예측값 | 본문 기본 결과 |
| 02 | `02_binary_reference/` | 기존 바이너리 정책의 실제 행동과 기대 행동 | 비교 기준·부록 |
| 03 | `03_institution_feature_ablation/` | 기관 E0·E1·E7 보상특징 비교 | 본문 기관 분석 |
| 04 | `04_institution_hyperparameters/` | 기관 E1의 규제·학습량 진단 | 부록·강건성 검증 |

각 하위 폴더는 다음을 포함한다.

- `fig_*.png`: GitHub 및 논문 삽입용 그림
- `data/`: 그림을 생성한 날짜별 집계 예측과 평가 수치
- `README.md`: 그림 설명, 해석 범위, 원본 경로

전체 파일의 원본 대응관계는 `SOURCE_MANIFEST.csv`에 기록했다.

## 핵심 그림 추천

1. 전체 투자자 비교: `01_continuous_all_investors/fig_01_all_investors_timeseries.png`
2. 기관 보상특징 효과: `03_institution_feature_ablation/fig_02_actual_vs_e0_e1_e7_rolling_20d.png`
3. 기관 실제값·예측값 산점도: `03_institution_feature_ablation/fig_03_actual_vs_e0_e1_e7_scatter.png`
4. 학습조건 진단: `04_institution_hyperparameters/fig_02_actual_vs_hyperparameters_rolling_20d.png`

## 관련 실험 기록

- 연속 순매수 기본 실험: `../../2026-07-08/2341_continuous_net_buy_behavior.md`
- 연속형·바이너리 비교: `../../2026-07-10/1627_continuous_vs_binary_evaluation.md`
- 기관 보상특징 실험: `../../2026-07-10/1718_continuous_institution_current_data_ablation.md`
- 기관 그래프 기록: `../../2026-07-10/1757_institution_actual_vs_prediction_graphs.md`
- 기관 학습조건 진단: `../../2026-07-10/2114_continuous_institution_hyperparameter_diagnostics.md`

## 주의

- CPCV에서 한 날짜는 여러 test path에 포함되므로 시계열 그림은 해당 날짜의 OOS 예측을 평균해 표시한다.
- 기관 모델은 방향성이 일부 개선됐지만 예측 진폭이 실제보다 크게 작다. 그림을 강도 복원 성공의 근거로 해석하지 않는다.
- `04_institution_hyperparameters`의 T2는 CPCV에서는 개선됐지만 초기 walk-forward 구간에서 과적합되어 주모형으로 채택하지 않았다.
