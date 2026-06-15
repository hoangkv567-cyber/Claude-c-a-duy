import ollama
import logging
from src.neo4j_client import Neo4jTCMClient
from src.utils import logger

class TCMQASystem:
    def __init__(self, config: dict = None):
        self.config = config or {}
        self.llm_model = self.config.get("qa", {}).get("llm_model", "qwen2.5:7b")
        
        # Sơ đồ graph mẫu để LLM dựa vào sinh câu truy vấn Cypher
        self.schema = """
        Nodes:
        - BenhLy {name: string}
        - HoiChung {name: string}
        - TrieuChung {name: string}
        - BaiThuoc {name: string}
        - ViThuoc {name: string}
        
        Relationships:
        - (:BenhLy)-[:CHIA_THÀNH]->(:HoiChung)
        - (:HoiChung)-[:CÓ_BIỂU_HIỆN]->(:TrieuChung)
        - (:HoiChung)-[:ĐƯỢC_ĐIỀU_TRỊ_BẰNG]->(:BaiThuoc)
        - (:BaiThuoc)-[:BAO_GỒM]->(:ViThuoc)
        """
        self.node_labels = ["BenhLy", "HoiChung", "TrieuChung", "BaiThuoc", "ViThuoc"]
        self.rel_types = ["CHIA_THÀNH", "CÓ_BIỂU_HIỆN", "ĐƯỢC_ĐIỀU_TRỊ_BẰNG", "BAO_GỒM"]
        
        self.neo4j_client = Neo4jTCMClient(
            uri=self.config.get("neo4j", {}).get("uri", "neo4j://localhost:7687"),
            user=self.config.get("neo4j", {}).get("user", "neo4j"),
            password=self.config.get("neo4j", {}).get("password", "12345678")
        )

    def text_to_cypher(self, question: str) -> str:
        prompt = f"""
You are an expert in translating natural language questions into Cypher queries for a Neo4j knowledge graph.

### GRAPH SCHEMA:
{self.schema}

### INSTRUCTIONS:
1. Output ONLY a valid Cypher query.
2. Do NOT include any explanations, markdown, or extra text.
3. Do NOT wrap the query in triple backticks or quotes.
4. Always use double quotes for string literals (e.g., "Tỳ vị hư nhược").
5. Node labels must be exactly: {', '.join(self.node_labels)}
6. Relationship types must be exactly: {', '.join(self.rel_types)}
7. Use the "name" property for all node names.

### USER QUESTION:
{question}

### CYPHER QUERY:
"""
        try:
            response = ollama.chat(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": "You are a precise Cypher query generator."},
                    {"role": "user", "content": prompt}
                ],
                options={
                    "temperature": self.config.get("qa", {}).get("temperature", 0.0),
                    "seed": self.config.get("qa", {}).get("seed", 42),
                    "top_p": 0.9
                }
            )
            cypher = response['message']['content'].strip()
            # Clean any markdown
            cypher = cypher.replace("```cypher", "").replace("```", "").strip()
            return cypher
        except Exception as e:
            logger.error(f"Lỗi sinh Cypher: {e}")
            return ""

    def answer_question(self, question: str) -> dict:
        """Dịch câu hỏi sang Cypher và truy vấn Neo4j để trả về kết quả"""
        cypher = self.text_to_cypher(question)
        if not cypher:
            return {"error": "Không thể dịch câu hỏi sang câu truy vấn database."}
        
        logger.info(f"Generated Cypher query: {cypher}")
        try:
            with self.neo4j_client.driver.session() as session:
                result = session.run(cypher)
                records = [record.data() for record in result]
                return {
                    "question": question,
                    "cypher_query": cypher,
                    "result": records
                }
        except Exception as e:
            logger.error(f"Lỗi thực thi Cypher: {e}")
            return {
                "question": question,
                "cypher_query": cypher,
                "error": f"Lỗi truy vấn cơ sở dữ liệu: {e}"
            }

    def close(self):
        """Đóng kết nối Neo4j"""
        self.neo4j_client.close()
