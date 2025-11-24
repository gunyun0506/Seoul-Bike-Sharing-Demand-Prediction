# 🚲 Seoul Bike Sharing Demand Prediction

Advanced Regression Modeling using Pseudo-Labeling & OOF Optimization

## 📌 Project Overview

이 프로젝트는 서울시 공공자전거(따릉이) 대여량을 예측하는 회귀(Regression) 분석 프로젝트입니다.
기상 데이터(기온, 습도, 강수량 등)와 시계열 정보를 활용하여 대여량을 예측하며, 초기 베이스라인 모델(RMSE 170+)에서 시작하여 고급 피처 엔지니어링과 앙상블 기법, 준지도 학습을 통해 최종 RMSE 105라는 Top-tier 성능을 달성했습니다. 기계학습 Kaggle Leaderbord에서 1등하였습니다.

## 🏆 Key Achievements

| Version | Strategy | RMSE Score | Improvement |
| :---: | :--- | :---: | :---: |
| **Baseline** | Linear Regression | 170+ | - |
| **v17** | LGBM Single Model + Feature Selection | 126 | ▲ 44 |
| **v30** | 4-Model Ensemble + Rolling Mean Features | 117 | ▲ 9 |
| **v42 (Final)** | **Pseudo-Labeling + OOF Optimization** | **105** | **▲ 12** |


## 💡 Core Strategies (문제 해결 노하우)

1. Advanced Feature Engineering (이동 평균 도입)

단순한 시차(Lag) 피처가 데이터 누수(Data Leakage)와 오염을 유발하여 RMSE 700점대 오류를 발생시켰던 문제를 해결하기 위해 이동 평균(Rolling Mean)을 도입했습니다.

shift(1): 1시간 전 날씨

rolling(3).mean(): 최근 3시간 평균 (단기 추세)

rolling(24).mean(): 최근 24시간 평균 (일일 추세)

Safety Lock: 데이터가 연속되지 않은 구간(00시, 누락 데이터)은 NaN 처리하여 학습 오염을 방지했습니다.

2. Dynamic Feature Selection (동적 특징 선택)

모델의 과적합을 방지하고 일반화 성능을 높이기 위해, XGBoost로 피처 중요도(Feature Importance)를 계산한 뒤 기여도가 가장 낮은 하위 3개의 피처를 자동으로 제거하는 동적 선택 방식을 적용했습니다.

3. OOF (Out-Of-Fold) Weight Optimization

단순 평균 앙상블이 아닌, 교차 검증 과정에서 도출된 OOF 예측값을 기반으로 RMSE를 최소화하는 가중치를 수학적으로 계산했습니다.

Discovery: 실험 결과 CatBoost가 압도적인 성능을 보임을 확인했습니다.

Optimized Weights: CatBoost (73%) + LightGBM (17%) + XGBoost (10%)

4. Pseudo-Labeling (의사 라벨링)

110점대에서 100점대로 진입한 핵심 기술입니다. Teacher-Student 구조를 사용하여 테스트 데이터의 분포를 학습했습니다.

Teacher Step: 최적화된 1단계 모델이 Test Set을 예측하여 '가짜 정답(Pseudo-Label)' 생성.

Student Step: Train Set + Pseudo-Labeled Test Set을 합쳐 데이터셋을 확장한 뒤 재학습.

## 📊 Project Workflow

![Seoul Bike Prediction Workflow](workflow.png)

## 🛠️ Environment & Libraries

Language: Python 3.x

Environment: Google Colab

Key Libraries:

pandas, numpy: 데이터 전처리

scikit-learn: KFold, RandomForest

xgboost, lightgbm, catboost: 핵심 예측 모델

scipy: 가중치 최적화

## 🚀 How to Run

# 1. Install dependencies
pip install xgboost lightgbm catboost pandas numpy scikit-learn

# 2. Prepare Data
Ensure 'train.csv' and 'test.csv' are in the correct directory.

# 3. Run the script
python main.ipynb


📂 File Structure

├── main.ipynb          # 전체 파이프라인 실행 코드
|
├── README.md           # 프로젝트 설명 문서
|
├── train.csv           # 학습 데이터
|
└── test.csv            # 테스트 데이터
