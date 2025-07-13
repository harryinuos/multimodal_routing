# find_path.py
import geopandas as gpd
import networkx as nx
import pandas as pd
import pickle
from shapely.geometry import Point

print("="*50)
print("경로탐색 스크립트를 시작합니다.")
print("="*50)

# --- 1. 설정: 파일 경로 및 탐색 파라미터 ---
# 전처리된 파일 경로
GRAPH_PICKLE_PATH = r'd:\multimodal_routing\convert_data\network_graph.pkl'
LINKS_FEATHER_PATH = r'd:\multimodal_routing\convert_data\links_data.feather'
NODES_FEATHER_PATH = r'd:\multimodal_routing\convert_data\nodes_data.feather'
CRS_FILE_PATH = r'd:\multimodal_routing\convert_data\crs.txt'

# 결과 저장 경로
OUTPUT_GEOJSON_PATH = r'd:\multimodal_routing\result\result_path.geojson'

# 출발지/도착지 좌표 (경도, 위도) - WGS84 (EPSG:4326)
start_coord = (127.0276, 37.4979) # 예: 강남역
end_coord = (126.9780, 37.5665)   # 예: 서울시청

# 사용할 컬럼명 (전처리 스크립트와 동일해야 함)
F_NODE_COL = 'F_NODE'
T_NODE_COL = 'T_NODE'
WEIGHT_COL = 'LENGTH'
LINK_ID_COL = 'LINK_ID'
NODE_ID_COL = 'NODE_ID'

# --- 2. 전처리된 데이터 로드 ---
try:
    print("\n[1/3] 저장된 그래프와 데이터 로드 중...")
    # 그래프 객체 로드
    with open(GRAPH_PICKLE_PATH, 'rb') as f:
        G = pickle.load(f)

    # 링크 데이터 로드
    links_df = pd.read_feather(LINKS_FEATHER_PATH)

    # 노드 데이터 로드
    nodes_df = pd.read_feather(NODES_FEATHER_PATH)
    # Feather 포맷은 geometry를 WKB로 저장하므로, GeoDataFrame으로 변환
    nodes_gdf = gpd.GeoDataFrame(nodes_df, geometry=gpd.GeoSeries.from_wkb(nodes_df['geometry']))

    # 좌표계 정보 로드
    with open(CRS_FILE_PATH, 'r') as f:
        crs_info = f.read()

    print(f"로드 완료: 노드 {G.number_of_nodes()}개, 엣지 {G.number_of_edges()}개")
except FileNotFoundError:
    print("오류: 전처리된 파일(.pkl, .feather)을 찾을 수 없습니다. 'create_graph.py'를 먼저 실행하세요.")
    exit()

# --- 3. 좌표 -> 최근접 노드 변환 ---
print("\n[2/4] 입력 좌표에서 가장 가까운 노드 탐색 중...")
nodes_gdf.crs = crs_info # 노드 데이터에 좌표계 정보 설정

def find_nearest_node(coord, nodes_gdf):
    """주어진 좌표(경도, 위도)에서 가장 가까운 노드를 찾습니다."""
    # 입력 좌표를 GeoSeries로 변환 (CRS: WGS84)
    point = gpd.GeoSeries([Point(coord)], crs='EPSG:4326')
    # 노드 데이터의 좌표계로 변환
    point_transformed = point.to_crs(nodes_gdf.crs)
    # 가장 가까운 노드 탐색
    distances = nodes_gdf.geometry.distance(point_transformed.iloc[0])
    nearest_node_idx = distances.idxmin()
    return nodes_gdf.loc[nearest_node_idx][NODE_ID_COL]

start_node = find_nearest_node(start_coord, nodes_gdf)
end_node = find_nearest_node(end_coord, nodes_gdf)
print(f"-> 출발 노드: {start_node} (입력: {start_coord})")
print(f"-> 도착 노드: {end_node} (입력: {end_coord})")

print(f"\n[3/4] 경로 탐색 중... ({start_node} -> {end_node})")

if start_node not in G or end_node not in G:
    print("[탐색 실패] 시작 또는 종료 노드가 그래프에 없습니다.")
    exit()

try:
    # 다익스트라 알고리즘으로 최단 경로 탐색
    path = nx.dijkstra_path(G, source=start_node, target=end_node, weight=WEIGHT_COL)
    length = nx.dijkstra_path_length(G, source=start_node, target=end_node, weight=WEIGHT_COL)
    print("탐색 성공!")

    # 경로(노드 리스트)를 기반으로 링크 ID 리스트 생성
    link_id_path = []
    for i in range(len(path) - 1):
        u, v = path[i], path[i+1]
        # DiGraph이므로 u -> v 방향으로만 탐색
        link_row = links_df[
            (links_df[F_NODE_COL] == u) & (links_df[T_NODE_COL] == v)
        ]
        if not link_row.empty:
            link_id_path.append(link_row.iloc[0][LINK_ID_COL])

    # --- 결과 GeoDataFrame 생성 및 저장 ---
    print("\n[4/4] 경로 결과를 GeoJSON 파일로 저장 중...")
    # 추출된 링크 ID로 경로에 해당하는 링크 필터링
    path_links_df = links_df[links_df[LINK_ID_COL].isin(link_id_path)].copy()

    # Feather 포맷은 geometry를 WKB(bytes)로 저장하므로, 이를 다시 geometry 객체로 변환해야 합니다.
    # 1. WKB 바이트를 geometry 객체로 변환합니다.
    path_geometry = gpd.GeoSeries.from_wkb(path_links_df['geometry'])
    # 2. 변환된 geometry를 사용하여 GeoDataFrame을 생성합니다.
    path_gdf = gpd.GeoDataFrame(path_links_df.drop(columns=['geometry']), geometry=path_geometry, crs=crs_info)

    # 시각화를 위해 경로 순서대로 정렬
    path_gdf[LINK_ID_COL] = pd.Categorical(path_gdf[LINK_ID_COL], categories=link_id_path, ordered=True)
    path_gdf = path_gdf.sort_values(LINK_ID_COL)

    # GeoJSON 파일로 저장
    path_gdf.to_file(OUTPUT_GEOJSON_PATH, driver='GeoJSON', encoding='utf-8')
    
    print("\n--- 최종 결과 ---")
    print(f"경로가 '{OUTPUT_GEOJSON_PATH}' 파일로 성공적으로 저장되었습니다.")
    print(f"총 거리: {length / 1000:.2f} km")
    # print(f"노드 경로: {path}") # 필요 시 주석 해제

except nx.NetworkXNoPath:
    print("\n[탐색 실패] 두 노드 사이에 경로가 존재하지 않습니다.")
except Exception as e:
    print(f"\n[오류] 경로 처리 중 문제가 발생했습니다: {e}")