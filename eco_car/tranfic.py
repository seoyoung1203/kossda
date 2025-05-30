import pandas as pd

# CSV 읽기 (깨진 경우 인코딩 확인 후 지정, 보통 cp949 또는 euc-kr)
df = pd.read_csv('data/수소전기차등록.csv', encoding='cp949')  

# UTF-8 (BOM 포함)로 저장 - 엑셀에서 한글 깨짐 방지
df.to_csv('수소전기차등록_.csv', encoding='utf-8-sig', index=False)