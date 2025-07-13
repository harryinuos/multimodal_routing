# find_path.py
import geopandas as gpd
import networkx as nx
import pandas as pd
import pickle
from shapely.geometry import Point, LineString
from shapely.ops import nearest_points, substring, linemerge
from itertools import product
import os
from create_graph import NETWORK_CONFIGS

print("="*50)
print("경로탐색 스크립트를 시작합니다.")
print("="*50)

# --- 1. 설정: 파일 경로 및 탐색 파라미터 ---
# 전처리된 파일 경로
# 스크립트 파일의 현재 위치
current_dir = os.path.dirname(os.path.abspath(__file__))
# 프로젝트 루트 폴더 (현재 폴더의 상위 폴더)
project_root = os.path.dirname(current_dir)

# 전처리된 데이터가 있는 폴더
CONVERT_DATA_DIR = os.path.join(project_root, 'convert_data')
# 결과가 저장될 폴더
RESULT_DIR = os.path.join(project_root, 'result')
os.makedirs(RESULT_DIR, exist_ok=True) # 결과 폴더가 없으면 생성

# --- 경로 탐색을 수행할 네트워크 선택 ---
NETWORK_NAME = 'walk_bike' # 'car', 'walk_bike' 등 'create_graph.py'에 설정된 'name'

# 선택된 네트워크의 설정(컬럼명 등) 가져오기
try:
    network_config = next(item for item in NETWORK_CONFIGS if item["name"] == NETWORK_NAME)
except StopIteration:
    print(f"오류: '{NETWORK_NAME}'에 대한 설정을 'create_graph.py'에서 찾을 수 없습니다.")
    exit()

# 사용할 컬럼명 (create_graph.py 설정에서 자동으로 가져옴)
cols = network_config['cols']
F_NODE_COL = cols['f_node']
T_NODE_COL = cols['t_node']
WEIGHT_COL = cols['weight']
LINK_ID_COL = cols['link_id']
NODE_ID_COL = cols['node_id']

# 파일 경로 (네트워크 이름에 따라 동적으로 생성)
GRAPH_PICKLE_PATH = os.path.join(CONVERT_DATA_DIR, f'{NETWORK_NAME}_network_graph.pkl')
LINKS_FEATHER_PATH = os.path.join(CONVERT_DATA_DIR, f'{NETWORK_NAME}_links_data.feather')
NODES_FEATHER_PATH = os.path.join(CONVERT_DATA_DIR, f'{NETWORK_NAME}_nodes_data.feather')
CRS_FILE_PATH = os.path.join(CONVERT_DATA_DIR, f'{NETWORK_NAME}_crs.txt')

# 결과 저장 경로
OUTPUT_GEOJSON_PATH = os.path.join(RESULT_DIR, f'result_path_{NETWORK_NAME}.geojson')

# 출발지/도착지 좌표 (경도, 위도) - WGS84 (EPSG:4326)
start_coord = (127.0276, 37.4979) # 예: 강남역
end_coord = (126.9780, 37.5665)   # 예: 서울시청

# 거리 계산의 정확도를 위한 투영 좌표계 (대한민국 중부원점 - EPSG:5186)
PROJECTED_CRS = "EPSG:5186"

# --- 2. 전처리된 데이터 로드 ---
try:
    print("\n[1/6] 저장된 그래프와 데이터 로드 중...")
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
    print(f"오류: '{NETWORK_NAME}' 네트워크의 전처리된 파일을 찾을 수 없습니다.")
    print(f"  - 확인 경로: {CONVERT_DATA_DIR}")
    print("  - 'create_graph.py'를 먼저 실행했는지, NETWORK_NAME이 올바른지 확인하세요.")
    exit()

# GeoDataFrame 생성 및 좌표계 설정
links_gdf = gpd.GeoDataFrame(links_df, geometry=gpd.GeoSeries.from_wkb(links_df['geometry']), crs=crs_info)

def find_nearest_link_and_projection(coord, links_gdf):
    """
    주어진 좌표(경도, 위도)에서 가장 가까운 링크와 그 링크 위의 투영점을 찾습니다.
    거리 계산의 정확도를 위해 내부적으로 투영 좌표계를 사용합니다.
    """
    # 입력 좌표를 GeoSeries로 변환 (CRS: WGS84)
    point = gpd.GeoSeries([Point(coord)], crs='EPSG:4326')
    
    # 링크와 포인트를 거리 계산에 적합한 투영 좌표계로 변환
    links_proj = links_gdf.to_crs(PROJECTED_CRS)
    point_proj = point.to_crs(PROJECTED_CRS).iloc[0]

    # 가장 가까운 링크 탐색 (투영 좌표계에서)
    distances = links_proj.geometry.distance(point_proj)
    nearest_link_idx = distances.idxmin()
    
    # 원본 links_gdf에서 가장 가까운 링크를 가져옴
    nearest_link = links_gdf.loc[nearest_link_idx]

    # 원본 좌표계에서 링크 위의 가장 가까운 지점(투영점) 계산
    point_transformed = point.to_crs(links_gdf.crs).iloc[0]
    projected_point = nearest_points(point_transformed, nearest_link.geometry)[1]

    return nearest_link, projected_point

# --- 3. 좌표 -> 최근접 링크 및 투영점 탐색 ---
print("\n[2/6] 입력 좌표에서 가장 가까운 링크 탐색 중...")
start_link, start_projection = find_nearest_link_and_projection(start_coord, links_gdf)
end_link, end_projection = find_nearest_link_and_projection(end_coord, links_gdf)

print(f"-> 출발 링크: {start_link[LINK_ID_COL]} (입력: {start_coord})")
print(f"-> 도착 링크: {end_link[LINK_ID_COL]} (입력: {end_coord})")

# --- 4. 경로 탐색 및 형상 재구성 ---
print("\n[3/6] 경로 유형 분석 중...")
final_path_gdf = None
total_length = 0

# Case 1: 출발지와 도착지가 동일한 링크에 있는 경우
if start_link[LINK_ID_COL] == end_link[LINK_ID_COL]:
    print("-> 출발지와 도착지가 동일 링크 상에 있습니다. 링크 내 경로를 생성합니다.")
    link_geom = start_link.geometry
    # project()는 라인 시작점부터의 거리를 반환
    start_dist = link_geom.project(start_projection)
    end_dist = link_geom.project(end_projection)

    # 두 점 사이의 라인 일부를 추출
    path_geom = substring(link_geom, min(start_dist, end_dist), max(start_dist, end_dist))
    total_length = path_geom.length
    final_path_gdf = gpd.GeoDataFrame([{'geometry': path_geom}], crs=crs_info)

# Case 2: 출발지와 도착지가 다른 링크에 있는 경우
else:
    print("-> 출발지와 도착지가 다른 링크 상에 있습니다. 최적 경로 조합을 탐색합니다.")
    
    # 후보 노드 및 부분 경로 거리 계산
    start_link_geom = start_link.geometry
    start_fnode, start_tnode = start_link[F_NODE_COL], start_link[T_NODE_COL]
    dist_proj_to_fnode = start_link_geom.project(start_projection)
    dist_proj_to_tnode = start_link_geom.length - dist_proj_to_fnode

    end_link_geom = end_link.geometry
    end_fnode, end_tnode = end_link[F_NODE_COL], end_link[T_NODE_COL]
    dist_proj_to_fnode_end = end_link_geom.project(end_projection)
    dist_proj_to_tnode_end = end_link_geom.length - dist_proj_to_fnode_end

    candidate_paths = []
    
    # 가능한 모든 노드 조합 (출발링크의 F/T 노드 -> 도착링크의 F/T 노드)
    start_nodes = [(start_fnode, dist_proj_to_fnode), (start_tnode, dist_proj_to_tnode)]
    end_nodes = [(end_fnode, dist_proj_to_fnode_end), (end_tnode, dist_proj_to_tnode_end)]

    for (s_node, s_dist), (e_node, e_dist) in product(start_nodes, end_nodes):
        try:
            # 그래프 상의 최단 경로 탐색
            path_nodes = nx.dijkstra_path(G, source=s_node, target=e_node, weight=WEIGHT_COL)
            path_len = nx.dijkstra_path_length(G, source=s_node, target=e_node, weight=WEIGHT_COL)
            
            total_len = s_dist + path_len + e_dist
            candidate_paths.append({
                'total_length': total_len,
                'path_nodes': path_nodes,
                'start_node_main': s_node,
                'end_node_main': e_node,
            })
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            continue

    if not candidate_paths:
        print("\n[탐색 실패] 후보 경로를 찾을 수 없습니다.")
        exit()

    # 가장 짧은 총 거리를 가진 경로 선택
    best_path_info = min(candidate_paths, key=lambda x: x['total_length'])
    total_length = best_path_info['total_length']
    main_path_nodes = best_path_info['path_nodes']

    print(f"\n[4/6] 최단 경로 탐색 완료 (총 거리: {total_length / 1000:.2f} km)")

    # --- 5. 최종 경로 형상 재구성 ---
    print("\n[5/6] 최종 경로 형상 재구성 중...")
    
    # 1. 시작 링크의 부분 형상
    start_proj_dist = start_link_geom.project(start_projection)
    if best_path_info['start_node_main'] == start_tnode: # 투영점 -> T_NODE 방향
        start_segment = substring(start_link_geom, start_proj_dist, start_link_geom.length)
    else: # 투영점 -> F_NODE 방향
        start_segment = substring(start_link_geom, 0, start_proj_dist)
        start_segment = LineString(list(start_segment.coords)[::-1]) # 방향 뒤집기

    # 2. 중간 경로의 전체 링크 형상
    middle_segments = []
    for i in range(len(main_path_nodes) - 1):
        u, v = main_path_nodes[i], main_path_nodes[i+1]
        
        # 경로 방향(u->v)과 일치하는 링크 검색
        link_row = links_gdf[(links_gdf[F_NODE_COL] == u) & (links_gdf[T_NODE_COL] == v)]
        if not link_row.empty:
            middle_segments.append(link_row.iloc[0].geometry)
            continue

        # 반대 방향(v->u) 링크 검색 (무방향 그래프용)
        link_row_reverse = links_gdf[(links_gdf[F_NODE_COL] == v) & (links_gdf[T_NODE_COL] == u)]
        if not link_row_reverse.empty:
            geom = link_row_reverse.iloc[0].geometry
            reversed_geom = LineString(list(geom.coords)[::-1]) # 방향 뒤집기
            middle_segments.append(reversed_geom)

    # 3. 도착 링크의 부분 형상
    end_proj_dist = end_link_geom.project(end_projection)
    if best_path_info['end_node_main'] == end_fnode: # F_NODE -> 투영점 방향
        end_segment = substring(end_link_geom, 0, end_proj_dist)
    else: # T_NODE -> 투영점 방향
        end_segment = substring(end_link_geom, end_proj_dist, end_link_geom.length)
        end_segment = LineString(list(end_segment.coords)[::-1]) # 방향 뒤집기

    # 4. 모든 형상 결합
    all_segments = [start_segment] + middle_segments + [end_segment]
    merged_path = linemerge(all_segments)
    
    final_path_gdf = gpd.GeoDataFrame([{'geometry': merged_path}], crs=crs_info)

# --- 6. 결과 저장 ---
if final_path_gdf is not None:
    print(f"\n[6/6] 경로 결과를 GeoJSON 파일로 저장 중...")
    final_path_gdf.to_file(OUTPUT_GEOJSON_PATH, driver='GeoJSON', encoding='utf-8')

    print("\n--- 최종 결과 ---")
    print(f"경로가 '{OUTPUT_GEOJSON_PATH}' 파일로 성공적으로 저장되었습니다.")
    print(f"총 거리: {total_length / 1000:.2f} km")
else:
    print("\n[오류] 최종 경로를 생성하지 못했습니다.")