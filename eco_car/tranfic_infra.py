import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 1. 데이터 불러오기
file_path = "교통인프라_분석결과.xlsx"
df = pd.read_excel(file_path)

# 2. 결측값 처리
required_cols = ['자전거보유수', '충전소수', '전기차수', '이산화질소_평균']
df = df.dropna(subset=required_cols)

# 3. 교통 인프라 지수 생성 (정규화 후 평균)
infra_cols = ['자전거보유수', '충전소수', '전기차수']
scaler = MinMaxScaler()
df[[f"{col}_정규화" for col in infra_cols]] = scaler.fit_transform(df[infra_cols])
df['교통인프라지수'] = df[[f"{col}_정규화" for col in infra_cols]].mean(axis=1)

# 4. 클러스터링
kmeans = KMeans(n_clusters=3, random_state=42)
df['클러스터'] = kmeans.fit_predict(df[['교통인프라지수']])

# 5. 상관관계 분석
corr = df[['전기차수', '충전소수', '자전거보유수', '교통인프라지수', '이산화질소_평균']].corr()

# 6. 히트맵(Heatmap) 시각화: 변수 간 상관관계
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap='YlGnBu', fmt='.2f')
plt.title('교통 인프라 및 대기오염 변수 간 상관관계 히트맵')
plt.show()

# 7. 밀도 플롯(Density Plot) 시각화: 교통인프라지수 vs 이산화질소_평균
plt.figure(figsize=(10, 6))
sns.kdeplot(
    data=df,
    x='교통인프라지수',
    y='이산화질소_평균',
    cmap='viridis',
    fill=True,
    thresh=0.05
)
plt.title('교통 인프라지수와 이산화질소 평균의 밀도 플롯')
plt.xlabel('교통인프라지수')
plt.ylabel('이산화질소_평균')
plt.show()
