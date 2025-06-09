import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['font.family'] = 'Malgun Gothic'  
plt.rcParams['axes.unicode_minus'] = False     # 마이너스 깨짐 방지

# 데이터 불러오기
df = pd.read_excel('교통인프라_인구비례분석결과.xlsx')

# 인구 기준 교통 인프라 지수 생성
infra_cols = ['인구1만명당_전기차', '인구1만명당_충전소', '인구1만명당_자전거']
scaler = MinMaxScaler()
df[[f'{col}_정규화' for col in infra_cols]] = scaler.fit_transform(df[infra_cols])
df['인구기준_교통인프라지수'] = df[[f'{col}_정규화' for col in infra_cols]].mean(axis=1)

# 클러스터링
kmeans = KMeans(n_clusters=3, random_state=42)
df['인구기준_클러스터'] = kmeans.fit_predict(df[['인구기준_교통인프라지수']])

# 상관관계 분석
corr = df[['인구1만명당_전기차', '인구1만명당_충전소', '인구1만명당_자전거', '인구기준_교통인프라지수', '이산화질소_평균']].corr()
print('\n[인구 기준 교통 인프라-대기오염 상관관계]')
print(corr['이산화질소_평균'])



# 커스텀 색상 지정 (클러스터 수에 맞게 3개)
custom_palette = ['#228B22', '#2E8B57', '#98FB98']

# pairplot 시각화
sns.pairplot(
    df,
    vars=['인구기준_교통인프라지수', '이산화질소_평균'],
    hue='인구기준_클러스터',
    palette=custom_palette
)

# 제목 추가
plt.suptitle('페어플롯: 인구 기준 교통 인프라 지수와 이산화질소 평균', y=1.02)
plt.show()

# plt.figure(figsize=(10,6))
# plt.scatter(df['인구기준_교통인프라지수'], df['이산화질소_평균'], c=df['인구기준_클러스터'], cmap='viridis', s=100)
# plt.xlabel('인구 기준 교통 인프라 지수')
# plt.ylabel('이산화질소_평균')
# plt.title('인구 기준 교통 인프라와 대기오염의 관계')
# plt.colorbar(label='클러스터')
# plt.grid(True)
# plt.show()