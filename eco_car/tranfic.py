

# ------------- 데이터 분석-----------------------------

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from functools import reduce

# 파일 경로
byc_path = "data/byc.csv"
pollution_path = "data/이산화질소.csv"
ev_path = "data/지역별 전기차.csv"
station_path = "data/버스 충전소.csv"

# 🚲 자전거 데이터
byc_df = pd.read_csv(byc_path)

# '시도별(1)' 컬럼명을 '시도'로 변경
byc_df.rename(columns={'시도별(1)': '시도'}, inplace=True)

byc_grouped = byc_df.groupby("시도")[["스테이션 개소 (개)", "자전거보유 (대)"]].sum().reset_index()
byc_grouped.columns = ["시도", "자전거스테이션수", "자전거보유수"]

# 🔋 전기차 수 데이터
elec_df = pd.read_csv(ev_path)

# wide -> long 변환
elec_long = elec_df.melt(id_vars="기준일", var_name="시도", value_name="전기차수")
# 지역명이 '서울' -> '서울특별시' 등의 시도 이름과 맞도록 변환 필요
region_map = {
    "서울": "서울특별시", "부산": "부산광역시", "대구": "대구광역시", "인천": "인천광역시",
    "광주": "광주광역시", "대전": "대전광역시", "울산": "울산광역시", "세종": "세종특별자치시",
    "경기": "경기도", "강원": "강원도", "충북": "충청북도", "충남": "충청남도",
    "전북": "전라북도", "전남": "전라남도", "경북": "경상북도", "경남": "경상남도",
    "제주": "제주특별자치도"
}

def normalize_region(region):
    return region_map.get(region, region)  # 매핑 없으면 원래 이름 리턴

# elec_df는 wide 형태로 (컬럼 고려)
elec_long = elec_df.melt(id_vars=['기준일'], var_name='시도', value_name='값')

# 함수 적용
elec_long["시도"] = elec_long["시도"].apply(normalize_region)

elec_grouped = elec_long.groupby("시도")["값"].sum().reset_index()
elec_grouped.columns = ["시도", "전기차수"]

# print(elec_df.columns.tolist())
# print(elec_long.columns.tolist())
# print(elec_long['시도'].unique())



# 충전소 수 데이터
charge_df = pd.read_csv(station_path)
charge_df = charge_df.rename(columns={"구분": "시도", charge_df.columns[1]: "충전소수"})

# 두 번째, 세 번째, 네 번째 컬럼을 더해서 '충전소수' 계산
cols_to_sum = charge_df.columns[1:4]  # 두 번째~네 번째 컬럼
charge_df["충전소수"] = charge_df[cols_to_sum].sum(axis=1)

# 시도별로 충전소수 집계 (혹시 동일 시도 중복 행 있을 경우 대비)
charge_grouped = charge_df.groupby("시도")["충전소수"].sum().reset_index()


charge_df = charge_df.groupby("시도")["충전소수"].sum().reset_index()

# 확인
# print(charge_df)

# 이산화질소 데이터

# 1. 미세먼지 데이터 불러오기
pollution_df = pd.read_csv(pollution_path)

# 2. 열 이름 정리 (공백 제거)
pollution_df.columns = pollution_df.columns.str.strip()

# 3. 필요한 컬럼 목록 (2024.05 ~ 2024.10)
target_cols = ["2024.05", "2024.06", "2024.07", "2024.08", "2024.09", "2024.10"]

# 4. 문자열 숫자를 정수로 변환 (쉼표 제거 후 numeric)
for col in target_cols:
    pollution_df[col] = pollution_df[col].astype(str).str.replace(",", "").str.replace(" ", "")
    pollution_df[col] = pd.to_numeric(pollution_df[col], errors="coerce")

# 5. 평균 계산 → '미세먼지양' 컬럼 생성
pollution_df["이산화질소소"] = pollution_df[target_cols].mean(axis=1)

# 6. 시도 기준으로 평균 집계 (구분(2)를 시도로)
pollution_avg = pollution_df.groupby("구분(2)")["이산화질소소"].mean().reset_index()
pollution_avg.columns = ["시도", "이산화질소_평균"]

# 7. 시도 이름 통일 (병합을 위해)
region_map = {
    "서울특별시": "서울", "부산광역시": "부산", "대구광역시": "대구", "인천광역시": "인천",
    "광주광역시": "광주", "대전광역시": "대전", "울산광역시": "울산", "세종특별자치시": "세종",
    "경기도": "경기", "강원도": "강원", "강원특별자치도": "강원", "충청북도": "충북", "충청남도": "충남",
    "전라북도": "전북", "전북특별자치도": "전북", "전라남도": "전남", "경상북도": "경북", "경상남도": "경남",
    "제주특별자치도": "제주"
}

pollution_avg["시도"] = pollution_avg["시도"].apply(lambda x: region_map.get(x.strip(), x.strip()))

# 8. 결과 확인
print(pollution_avg)


def normalize_region(region):
    region = region.strip()
    return region_map.get(region, region)

# 각 데이터프레임에 적용
byc_grouped["시도"] = byc_grouped["시도"].apply(normalize_region)
elec_grouped["시도"] = elec_grouped["시도"].apply(normalize_region)
charge_df["시도"] = charge_df["시도"].apply(normalize_region)
pollution_avg["시도"] = pollution_avg["시도"].apply(normalize_region)


# #------------------전처리 끝 ---------------------------------------------

# 병합 -> 시도 컬럼 기준으로
dfs = [byc_grouped, elec_grouped, charge_df, pollution_avg]
merged_df = reduce(lambda left, right: pd.merge(left, right, on="시도", how="outer"), dfs)
# 통합 df

# 병합 결과 확인
# print("\n[병합된 데이터프레임]")


# 결측값 제거
merged_df.dropna(inplace=True)

# 만약 안되면 경고 출력
if merged_df.empty:
    raise ValueError("병합 결과가 비어 있습니다. 시도명이 맞는지 확인하세요.")

print(merged_df)


# 교통 인프라 지수 (정규화 후 평균 냄냄)
infra_cols = ["자전거보유수", "충전소수", "전기차수"]
scaler = MinMaxScaler()
merged_df[[f"{col}_정규화" for col in infra_cols]] = scaler.fit_transform(merged_df[infra_cols])
merged_df["교통인프라지수"] = merged_df[[f"{col}_정규화" for col in infra_cols]].mean(axis=1)


# 클러스터링 - 교통 인프라 수준이 비슷한 지역끼리 자동으로 묶어줌
kmeans = KMeans(n_clusters=3, random_state=42)
merged_df["클러스터"] = kmeans.fit_predict(merged_df[["교통인프라지수"]])

# 상관관계 분석
corr = merged_df[["전기차수", "충전소수", "자전거보유수", "교통인프라지수", "이산화질소_평균"]].corr()

# 출력 결과
print("\n[시도별 교통 인프라 요약]")
print(merged_df[["시도", "교통인프라지수", "클러스터", "이산화질소_평균"]])

print("\n[전기차 및 대기오염 상관관계]")
print(corr[["이산화질소_평균"]].sort_values(by="이산화질소_평균"))

# 저장
merged_df.to_excel("교통인프라_분석결과.xlsx", index=False)

