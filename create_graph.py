# create_graph.py
import geopandas as gpd
import networkx as nx
import pandas as pd
import pickle
import os

print("="*50)
print("전처리 스크립트를 시작합니다.")
print("="*50)

# --- 1. 설정: 네트워크별 설정 및 경로 ---
# 스크립트 파일의 현재 위치
current_dir = os.path.dirname(os.path.abspath(__file__))
# 프로젝트 루트 폴더 (현재 폴더의 상위 폴더)
project_root = os.path.dirname(current_dir)

# 원본 데이터가 있는 폴더 (프로젝트 루트/raw_data)
RAW_DATA_DIR = os.path.join(project_root, 'raw_data')
# 전처리된 데이터가 저장될 폴더
CONVERT_DATA_DIR = os.path.join(project_root, 'convert_data')

# 네트워크 설정
# 'name': 결과 파일명에 사용될 네트워크 이름
# 'link_shp': 링크 SHP 파일 경로
# 'node_shp': 노드 SHP 파일 경로
# 'directed': True이면 방향 그래프(DiGraph), False이면 무방향 그래프(Graph) 생성
# 'cols': 네트워크 데이터에 사용되는 컬럼명
NETWORK_CONFIGS = [
    {
        'name': 'car',
        'link_shp': os.path.join(RAW_DATA_DIR, r'[2025-05-12]NODELINKDATA\MOCT_LINK.shp'),
        'node_shp': os.path.join(RAW_DATA_DIR, r'[2025-05-12]NODELINKDATA\MOCT_NODE.shp'),
        'directed': True,
        'encoding': 'cp949', # SHP 파일 인코딩
        'cols': {
            'f_node': 'F_NODE',
            't_node': 'T_NODE',
            'weight': 'LENGTH',
            'node_id': 'NODE_ID',
            'link_id': 'LINK_ID',
        }
    },
    {
        'name': 'walk_bike',
        # 아래 경로는 예시이며, 실제 도보/자전거 네트워크 파일 경로로 수정해야 합니다.
        'link_shp': os.path.join(RAW_DATA_DIR, r'seoul_ped_network\seoul_ped_network_link_5186.shp'),
        'node_shp': os.path.join(RAW_DATA_DIR, r'seoul_ped_network\seoul_ped_network_node_5186.shp'),
        'directed': False, # 무방향 그래프
        'encoding': 'cp949', # SHP 파일 인코딩 (예: 'utf-8', 'cp949')
        'cols': {
            # 예시: 도보/자전거 데이터의 컬럼명이 다른 경우 이 부분을 수정하세요.
            'f_node': 'st_nd_id',
            't_node': 'ed_nd_id',
            'weight': 'length',
            'node_id': 'node_id',
            'link_id': 'link_id',
        }
    }
]

def preprocess_network(config, output_dir):
    """
    주어진 설정에 따라 네트워크 데이터를 전처리하고 그래프를 생성하여 저장합니다.
    """
    network_name = config['name']
    link_shp_path = config['link_shp']
    node_shp_path = config['node_shp']
    is_directed = config['directed']
    encoding = config.get('encoding', 'utf-8') # 설정에서 인코딩 가져오기, 없으면 utf-8 기본값
    cols = config['cols']

    print(f"\n--- '{network_name}' 네트워크 처리 시작 ---")

    # --- 2. 데이터 로드 및 정제 ---
    try:
        print(f"[{network_name} 1/4] SHP 파일 로드 중 (인코딩: {encoding})...")
        links_gdf = gpd.read_file(link_shp_path, encoding=encoding)
        nodes_gdf = gpd.read_file(node_shp_path, encoding=encoding)
        print("SHP 파일 로드 완료.")
    except Exception as e:
        print(f"오류: '{network_name}' 네트워크의 SHP 파일 로드 실패. 경로를 확인하세요.")
        print(f"  - 링크: {link_shp_path}")
        print(f"  - 노드: {node_shp_path}")
        print(f"  - 상세: {e}")
        return # 다음 네트워크 처리로 넘어감

    print(f"[{network_name} 2/4] 데이터 정제 중...")
    
    # 지오메트리 유효성 검사 및 정제 (강화된 방식)
    initial_link_count = len(links_gdf)

    # 1. Null, 빈(empty) 지오메트리 제거
    # 2. 유효하지 않은(invalid) 지오메트리 제거 (예: self-intersecting lines)
    is_good_geom = links_gdf.geometry.notna() & ~links_gdf.geometry.is_empty & links_gdf.geometry.is_valid
    
    if not is_good_geom.all():
        links_gdf = links_gdf[is_good_geom]
        cleaned_count = initial_link_count - len(links_gdf)
        print(f"-> 유효하지 않거나(invalid) 비어있는 지오메트리를 가진 링크 {cleaned_count}개를 제거했습니다.")

    links_gdf[cols['f_node']] = pd.to_numeric(links_gdf[cols['f_node']], errors='coerce')
    links_gdf[cols['t_node']] = pd.to_numeric(links_gdf[cols['t_node']], errors='coerce')
    nodes_gdf[cols['node_id']] = pd.to_numeric(nodes_gdf[cols['node_id']], errors='coerce')
    links_gdf.dropna(subset=[cols['f_node'], cols['t_node']], inplace=True)
    print("데이터 정제 완료.")

    # --- 3. 그래프 생성 ---
    print(f"[{network_name} 3/4] NetworkX 그래프 생성 중...")
    graph_type = nx.DiGraph if is_directed else nx.Graph
    graph_type_name = "방향성" if is_directed else "무방향성"
    print(f"-> ({graph_type_name} 그래프)")

    G = nx.from_pandas_edgelist(links_gdf, source=cols['f_node'], target=cols['t_node'], edge_attr=[cols['weight'], cols['link_id']], create_using=graph_type)
    G.add_nodes_from(nodes_gdf[cols['node_id']])
    print(f"그래프 생성 완료: 노드 {G.number_of_nodes()}개, 엣지 {G.number_of_edges()}개")

    # --- 4. 전처리된 데이터 파일로 저장 ---
    print(f"[{network_name} 4/4] 생성된 그래프와 데이터를 파일로 저장 중...")
    
    # 출력 파일 경로 생성
    graph_pickle_path = os.path.join(output_dir, f'{network_name}_network_graph.pkl')
    links_feather_path = os.path.join(output_dir, f'{network_name}_links_data.feather')
    nodes_feather_path = os.path.join(output_dir, f'{network_name}_nodes_data.feather')
    crs_file_path = os.path.join(output_dir, f'{network_name}_crs.txt')

    # 4-1. 그래프 객체 저장 (pickle)
    with open(graph_pickle_path, 'wb') as f:
        pickle.dump(G, f)
    print(f"-> 그래프 저장 완료: {graph_pickle_path}")

    # 4-2. 링크 데이터 저장 (feather)
    links_for_path = links_gdf[[cols['link_id'], cols['f_node'], cols['t_node'], 'geometry']].copy()
    links_for_path.to_feather(links_feather_path)
    print(f"-> 링크 데이터 저장 완료: {links_feather_path}")

    # 4-3. 노드 데이터 저장 (feather)
    nodes_for_path = nodes_gdf[[cols['node_id'], 'geometry']].copy()
    nodes_for_path.to_feather(nodes_feather_path)
    print(f"-> 노드 데이터 저장 완료: {nodes_feather_path}")

    # 4-4. 좌표계(CRS) 정보 저장
    with open(crs_file_path, 'w') as f:
        f.write(links_gdf.crs.to_string())
    print(f"-> 좌표계 정보 저장 완료: {crs_file_path}")

if __name__ == "__main__":
    # 결과 저장 폴더가 없으면 생성
    os.makedirs(CONVERT_DATA_DIR, exist_ok=True)

    # 설정된 모든 네트워크에 대해 전처리 실행
    for config in NETWORK_CONFIGS:
        preprocess_network(config, CONVERT_DATA_DIR)

    print("\n" + "="*50)
    print("모든 네트워크 전처리 과정이 성공적으로 완료되었습니다.")
    print("="*50)
