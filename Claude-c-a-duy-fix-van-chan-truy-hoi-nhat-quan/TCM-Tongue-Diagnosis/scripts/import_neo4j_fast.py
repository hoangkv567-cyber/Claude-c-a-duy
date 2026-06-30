import sys
import os
sys.path.append("C:/Users/hoang/TTDN/TCM-Qwen-QA")
sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
from neo4j import GraphDatabase
import yaml

def main():
    with open("config/config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    neo4j_cfg = config.get("neo4j", {})
    uri = neo4j_cfg.get("uri", "neo4j+s://c55f875f.databases.neo4j.io")
    user = neo4j_cfg.get("user", "c55f875f")
    password = neo4j_cfg.get("password", "Z7b-auwCd7T1KPY8TF0p3_piWcAyfospK55nC196c7w")
    csv_path = config.get("dataset", {}).get("csv_path", "data/Medicine_clean.csv")

    print(f"Kết nối tới Neo4j tại: {uri}")
    driver = GraphDatabase.driver(uri, auth=(user, password))
    
    print(f"Đang đọc dữ liệu từ: {csv_path}")
    df = pd.read_csv(csv_path, encoding='utf-8')
    
    print("Xóa toàn bộ dữ liệu cũ trong Neo4j...")
    with driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")

    # Chuẩn bị các lô dữ liệu (batches)
    benh_hoichung_batch = []
    hoichung_baithuoc_batch = []
    hoichung_trieuchung_batch = []
    baithuoc_vithuoc_batch = []

    for _, row in df.iterrows():
        benh = row['tên_bệnh']
        hoichung = row['hội_chứng']
        bai_thuoc = row['bài_thuốc']
        
        benh_hoichung_batch.append({'benh': benh, 'hoichung': hoichung})
        hoichung_baithuoc_batch.append({'hoichung': hoichung, 'bai_thuoc': bai_thuoc, 'benh': benh})
        
        if pd.notna(row['triệu_chứng']):
            for symptom in [s.strip() for s in row['triệu_chứng'].split(',') if s.strip()]:
                hoichung_trieuchung_batch.append({'hoichung': hoichung, 'symptom': symptom, 'benh': benh})
                
        if pd.notna(row['vị_thuốc']):
            for herb in [h.strip() for h in row['vị_thuốc'].split(',') if h.strip()]:
                baithuoc_vithuoc_batch.append({'bai_thuoc': bai_thuoc, 'herb': herb, 'benh': benh, 'hoichung': hoichung})

    print(f"Số lượng quan hệ Bệnh lý - Hội chứng: {len(benh_hoichung_batch)}")
    print(f"Số lượng quan hệ Hội chứng - Bài thuốc: {len(hoichung_baithuoc_batch)}")
    print(f"Số lượng quan hệ Hội chứng - Triệu chứng: {len(hoichung_trieuchung_batch)}")
    print(f"Số lượng quan hệ Bài thuốc - Vị thuốc: {len(baithuoc_vithuoc_batch)}")

    # Thực thi các truy vấn bulk import bằng UNWIND
    with driver.session() as session:
        print("Đang nhập Bệnh lý & Hội chứng...")
        session.run("""
            UNWIND $batch AS row
            MERGE (b:BenhLy {name: row.benh})
            MERGE (h:HoiChung {name: row.hoichung})
            MERGE (b)-[:CHIA_THÀNH]->(h)
        """, batch=benh_hoichung_batch).consume()

        print("Đang nhập Bài thuốc...")
        session.run("""
            UNWIND $batch AS row
            MATCH (h:HoiChung {name: row.hoichung})
            MERGE (p:BaiThuoc {name: row.bai_thuoc, benh_ly: row.benh, hoi_chung: row.hoichung})
            MERGE (h)-[:ĐƯỢC_ĐIỀU_TRỊ_BẰNG]->(p)
        """, batch=hoichung_baithuoc_batch).consume()

        print("Đang nhập Triệu chứng...")
        chunk_size = 1000
        for i in range(0, len(hoichung_trieuchung_batch), chunk_size):
            chunk = hoichung_trieuchung_batch[i:i+chunk_size]
            session.run("""
                UNWIND $batch AS row
                MATCH (h:HoiChung {name: row.hoichung})
                MERGE (t:TrieuChung {name: row.symptom})
                MERGE (h)-[r:CÓ_BIỂU_HIỆN {benh_ly: row.benh}]->(t)
            """, batch=chunk).consume()

        print("Đang nhập Vị thuốc...")
        for i in range(0, len(baithuoc_vithuoc_batch), chunk_size):
            chunk = baithuoc_vithuoc_batch[i:i+chunk_size]
            session.run("""
                UNWIND $batch AS row
                MATCH (p:BaiThuoc {name: row.bai_thuoc, benh_ly: row.benh, hoi_chung: row.hoichung})
                MERGE (v:ViThuoc {name: row.herb})
                MERGE (p)-[:BAO_GỒM]->(v)
            """, batch=chunk).consume()

    driver.close()
    print("Bulk import hoàn tất thành công!")

if __name__ == '__main__':
    main()
