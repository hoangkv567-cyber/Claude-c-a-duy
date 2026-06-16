from neo4j import GraphDatabase
import logging

logger = logging.getLogger(__name__)

class Neo4jTCMClient:
    def __init__(self, uri="neo4j+s://c55f875f.databases.neo4j.io", user="c55f875f", password="Z7b-auwCd7T1KPY8TF0p3_piWcAyfospK55nC196c7w"):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        logger.info(f"Kết nối Neo4j thành công: {uri}")

    def close(self):
        self.driver.close()
        logger.info("Đã đóng kết nối Neo4j")

    def get_treatment_by_syndrome(self, syndrome_name: str) -> dict:
        """Lấy bài thuốc và vị thuốc theo hội chứng"""
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (s:HoiChung {name: $name})-[:ĐƯỢC_ĐIỀU_TRỊ_BẰNG]->(p:BaiThuoc)
                OPTIONAL MATCH (p)-[:BAO_GÔM]->(v:ViThuoc)
                RETURN s.name AS hoi_chung, 
                       p.name AS bai_thuoc, 
                       COLLECT(DISTINCT v.name) AS vi_thuoc
                LIMIT 1
                """,
                name=syndrome_name
            )
            record = result.single()
            if record:
                return {
                    "hoi_chung": record["hoi_chung"],
                    "bai_thuoc": record["bai_thuoc"],
                    "vi_thuoc": record["vi_thuoc"]
                }
            return None

    def get_all_syndromes(self) -> list:
        """Lấy danh sách tất cả hội chứng"""
        with self.driver.session() as session:
            result = session.run(
                "MATCH (s:HoiChung) RETURN s.name AS name ORDER BY name"
            )
            return [record["name"] for record in result]

    def get_symptoms_by_syndrome(self, syndrome_name: str) -> list:
        """Lấy danh sách triệu chứng theo hội chứng"""
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (s:HoiChung {name: $name})-[:CÓ_BIỂU_HIỆN]->(t:TrieuChung)
                RETURN t.name AS trieu_chung
                """,
                name=syndrome_name
            )
            return [record["trieu_chung"] for record in result]

    def get_diseases_by_syndrome(self, syndrome_name: str) -> list:
        """Lấy danh sách bệnh có hội chứng này"""
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (b:BenhLy)-[:CHIA_THÀNH]->(s:HoiChung {name: $name})
                RETURN b.name AS benh_ly
                """,
                name=syndrome_name
            )
            return [record["benh_ly"] for record in result]