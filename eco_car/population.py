import pandas as pd

# 1. 인구 데이터 불러오기
pop_path = '행정구역별_인구수.csv'
df_pop = pd.read_csv(pop_path, encoding='utf-8-sig')

# 2. '총인구수 (명)'으로 시작하는 모든 컬럼 추출
pop_cols = [col for col in df_pop.columns if col.startswith('총인구수 (명)')]

# 3. 각 행별로 해당 컬럼들의 평균 계산
df_pop['총인구수_평균'] = df_pop[pop_cols].mean(axis=1)

# 4. 시도명과 평균 인구수만 추출
df_pop_cleaned = df_pop[['행정구역(시군구)별', '총인구수_평균']].copy()
df_pop_cleaned.columns = ['시도', '총인구수_평균']

# 5. 시도명 표준화(교통 인프라 데이터와 병합을 위해)
region_map = {
    "서울특별시": "서울", "부산광역시": "부산", "대구광역시": "대구", "인천광역시": "인천",
    "광주광역시": "광주", "대전광역시": "대전", "울산광역시": "울산", "세종특별자치시": "세종",
    "경기도": "경기", "강원특별자치도": "강원", "강원도": "강원", "충청북도": "충북", "충청남도": "충남",
    "전북특별자치도": "전북", "전라북도": "전북", "전라남도": "전남", "경상북도": "경북", "경상남도": "경남",
    "제주특별자치도": "제주", "제주도": "제주"
}
df_pop_cleaned['시도'] = df_pop_cleaned['시도'].apply(lambda x: region_map.get(x.strip(), x.strip()))

# 6. 교통 인프라 데이터 불러오기
infra_df = pd.read_excel('교통인프라_분석결과.xlsx')  # 파일명에 맞게 수정

# 7. 병합
merged_df = pd.merge(infra_df, df_pop_cleaned, on='시도', how='left')

# 8. 인구 1만 명당 인프라 지표 계산
merged_df['인구1만명당_전기차'] = merged_df['전기차수'] / (merged_df['총인구수_평균'] / 10000)
merged_df['인구1만명당_충전소'] = merged_df['충전소수'] / (merged_df['총인구수_평균'] / 10000)
merged_df['인구1만명당_자전거'] = merged_df['자전거보유수'] / (merged_df['총인구수_평균'] / 10000)

# 9. 결과 확인
print(merged_df[['시도', '전기차수', '충전소수', '자전거보유수', '총인구수_평균', '인구1만명당_전기차', '인구1만명당_충전소', '인구1만명당_자전거']])

merged_df.to_excel("교통인프라_인구비례분석결과.xlsx", index=False)