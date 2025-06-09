import pandas as pd
import plotly.express as px
from dash import Dash, html, dcc

# 데이터 불러오기
file_path = '교통인프라_분석결과.xlsx'
df = pd.read_excel(file_path, sheet_name='Sheet1')

# 클러스터 이름 매핑 (옵션)
cluster_names = {0: 'Cluster 0', 1: 'Cluster 1', 2: 'Cluster 2'}
df['클러스터명'] = df['클러스터'].map(cluster_names)

# 산점도 그래프 (교통인프라지수 vs 이산화질소_평균)
scatter_fig = px.scatter(
    df, x='교통인프라지수', y='이산화질소_평균', color='클러스터명',
    hover_data=['시도'],
    title='교통 인프라 지수 vs 대기오염(이산화질소)',
    labels={'교통인프라지수': '교통 인프라 지수', '이산화질소_평균': '이산화질소 평균'}
)

# 막대그래프 (클러스터별 평균 대기오염)
bar_data = df.groupby('클러스터명')['이산화질소_평균'].mean().reset_index()
bar_fig = px.bar(
    bar_data, x='클러스터명', y='이산화질소_평균',
    title='클러스터별 평균 대기오염(이산화질소)',
    labels={'클러스터명': '클러스터', '이산화질소_평균': '평균 이산화질소'}
)

# Dash 앱 생성
app = Dash(__name__)
app.layout = html.Div([
    html.H1('교통 인프라와 대기오염 대시보드'),
    dcc.Graph(id='scatter-plot', figure=scatter_fig),
    dcc.Graph(id='bar-chart', figure=bar_fig),
    html.H2('데이터 테이블'),
    html.Table([
        # 테이블 헤더
        html.Thead(html.Tr([html.Th(col) for col in df.columns if col != '클러스터명'])),
        # 테이블 바디
        html.Tbody([
            html.Tr([html.Td(df.iloc[i][col]) for col in df.columns if col != '클러스터명']) 
            for i in range(len(df))
        ])
    ])
])

if __name__ == '__main__':
    app.run(debug=True)

