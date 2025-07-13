# create_graph.py
import geopandas as gpd
import networkx as nx
import pandas as pd
import pickle

print("="*50)
print("전처리 스크립트를 시작합니다. (최초 1회 실행)")
print("="*50)

# --- 1. 설정: 파일 경로 및 컬럼명 ---
LINK_SHP_PATH = r'd:\multimodal_routing\raw_data\[2025-05-12]NODELINKDATA\MOCT_LINK.shp'
NODE_SHP_PATH = r'd:\multimodal_routing\raw_data\[2025-05-12]NODELINKDATA\MOCT_NODE.shp'

# 저장될 파일 경로
GRAPH_PICKLE_PATH = r'd:\multimodal_routing\convert_data\network_graph.pkl'
LINKS_FEATHER_PATH = r'd:\multimodal_routing\convert_data\links_data.feather'
CRS_FILE_PATH = r'd:\multimodal_routing\convert_data\crs.txt'

# 사용할 컬럼명
F_NODE_COL = 'F_NODE'
T_NODE_COL = 'T_NODE'
WEIGHT_COL = 'LENGTH'
NODE_ID_COL = 'NODE_ID'
LINK_ID_COL = 'LINK_ID'

# --- 2. 데이터 로드 및 정제 ---
try:
    print(f"\n[1/4] SHP 파일 로드 중...")
    links_gdf = gpd.read_file(LINK_SHP_PATH, encoding='cp949')
    nodes_gdf = gpd.read_file(NODE_SHP_PATH, encoding='cp949')
    print("SHP 파일 로드 완료.")
except Exception as e:
    print(f"오류: SHP 파일 로드 실패. \n({e})")
    exit()

print("\n[2/4] 데이터 정제 중...")
links_gdf[F_NODE_COL] = pd.to_numeric(links_gdf[F_NODE_COL], errors='coerce')
links_gdf[T_NODE_COL] = pd.to_numeric(links_gdf[T_NODE_COL], errors='coerce')
nodes_gdf[NODE_ID_COL] = pd.to_numeric(nodes_gdf[NODE_ID_COL], errors='coerce')
links_gdf.dropna(subset=[F_NODE_COL, T_NODE_COL], inplace=True)
print("데이터 정제 완료.")

# --- 3. 그래프 생성 ---
print("\n[3/4] NetworkX 그래프 생성 중...")
# DiGraph를 사용하여 방향성(일방통행)을 고려한 그래프 생성
G = nx.from_pandas_edgelist(links_gdf, source=F_NODE_COL, target=T_NODE_COL, edge_attr=[WEIGHT_COL, LINK_ID_COL], create_using=nx.DiGraph)
G.add_nodes_from(nodes_gdf[NODE_ID_COL])
print(f"그래프 생성 완료: 노드 {G.number_of_nodes()}개, 엣지 {G.number_of_edges()}개")

# --- 4. 전처리된 데이터 파일로 저장 ---
print("\n[4/4] 생성된 그래프와 데이터를 파일로 저장 중...")
# 4-1. 그래프 객체 저장 (pickle)
with open(GRAPH_PICKLE_PATH, 'wb') as f:
    pickle.dump(G, f)
print(f"-> 그래프 저장 완료: {GRAPH_PICKLE_PATH}")

# 4-2. 링크 데이터 저장 (feather)
# GeoDataFrame에서 geometry를 제외한 필수 정보만 feather로 저장하여 속도 향상
links_for_path = links_gdf[[LINK_ID_COL, F_NODE_COL, T_NODE_COL, 'geometry']].copy()
links_for_path.to_feather(LINKS_FEATHER_PATH)
print(f"-> 링크 데이터 저장 완료: {LINKS_FEATHER_PATH}")

# 4-3. 좌표계(CRS) 정보 저장
with open(CRS_FILE_PATH, 'w') as f:
    f.write(links_gdf.crs.to_string())
print(f"-> 좌표계 정보 저장 완료: {CRS_FILE_PATH}")

print("\n모든 전처리 과정이 성공적으로 완료되었습니다.")