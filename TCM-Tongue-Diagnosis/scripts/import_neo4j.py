import pandas as pd
from neo4j import GraphDatabase
from src.config_loader import load_config

def create_graph(tx, row):
    # 1. Tạo BenhLy và HoiChung với quan hệ CHIA_THÀNH
    tx.run("""
        MERGE (b:BenhLy {name: $benh})
        MERGE (h:HoiChung {name: $hoichung})
        MERGE (b)-[:CHIA_THÀNH]->(h)
    """, benh=row['tên_bệnh'], hoichung=row['hội_chứng'])
    
    # 2. Tạo BaiThuoc và quan hệ ĐƯỢC_ĐIỀU_TRỊ_BẰNG
    tx.run("""
        MERGE (h:HoiChung {name: $hoichung})
        MERGE (p:BaiThuoc {name: $bai_thuoc})
        MERGE (h)-[:ĐƯỢC_ĐIỀU_TRỊ_BẰNG]->(p)
    """, hoichung=row['hội_chứng'], bai_thuoc=row['bài_thuốc'])
    
    # 3. Tạo TrieuChung và quan hệ CÓ_BIỂU_HIỆN
    if pd.notna(row['triệu_chứng']):
        for symptom in [s.strip() for s in row['triệu_chứng'].split(',')]:
            tx.run("""
                MATCH (h:HoiChung {name: $hoichung})
                MERGE (t:TrieuChung {name: $symptom})
                MERGE (h)-[:CÓ_BIỂU_HIỆN]->(t)
            """, hoichung=row['hội_chứng'], symptom=symptom)
    
    # 4. Tạo ViThuoc và quan hệ BAO_GÔM
    if pd.notna(row['vị_thuốc']):
        for herb in [h.strip() for h in row['vị_thuốc'].split(',')]:
            tx.run("""
                MATCH (p:BaiThuoc {name: $bai_thuoc})
                MERGE (v:ViThuoc {name: $herb})
                MERGE (p)-[:BAO_GÔM]->(v)
            """, bai_thuoc=row['bài_thuốc'], herb=herb)

def main():
    config = load_config("config/config.yaml")
    neo4j_cfg = config.get("neo4j", {})
    uri = neo4j_cfg.get("uri", "neo4j://localhost:7687")
    user = neo4j_cfg.get("user", "neo4j")
    password = neo4j_cfg.get("password", "12345678")
    csv_path = config.get("dataset", {}).get("csv_path", "data/tcm_data_600_clean.csv")

    driver = GraphDatabase.driver(uri, auth=(user, password))
    df = pd.read_csv(csv_path, encoding='utf-8')
    
    # Optional: Clear existing graph data to avoid duplication/conflicts
    print("Xoa du lieu cu trong Neo4j...")
    with driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")
        
    print(f"Bat dau import tu {csv_path}...")
    with driver.session() as session:
        for idx, row in df.iterrows():
            session.execute_write(create_graph, row)
    driver.close()
    print("Import hoan tat!")

if __name__ == "__main__":
    main()

