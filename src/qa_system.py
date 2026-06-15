# src/qa_system.py
import json
import logging
import re
from neo4j import GraphDatabase
import ollama
from src.config_loader import load_config

logger = logging.getLogger(__name__)

class TCMQA:
    def __init__(self, config: dict = None):
        self.config = config or load_config()
        self.llm_model = self.config.get("qa", {}).get("llm_model", "qwen2.5:7b")
        self.temperature = self.config.get("qa", {}).get("temperature", 0.0)
        self.seed = self.config.get("qa", {}).get("seed", 42)
        self.top_p = self.config.get("qa", {}).get("top_p", 0.9)

        # Kết nối Neo4j
        neo4j_cfg = self.config.get("neo4j", {})
        self.driver = GraphDatabase.driver(
            neo4j_cfg.get("uri", "neo4j://localhost:7687"),
            auth=(neo4j_cfg.get("user", "neo4j"), neo4j_cfg.get("password", "12345678"))
        )
        logger.info(f"Đã kết nối Neo4j: {neo4j_cfg.get('uri')}")

        # Đọc schema
        with open("data/graph_schema.txt", "r", encoding="utf-8") as f:
            self.schema = f.read()
        self.db_schema = self.schema
        logger.info(f"Đã tải schema từ data/graph_schema.txt")

        # Danh sách nhãn và quan hệ để LLM biết
        self.node_labels = ["BenhLy", "HoiChung", "TrieuChung", "BaiThuoc", "ViThuoc"]
        self.rel_types = ["CHIA_THANH", "CO_BIEU_HIEN", "DUOC_DIEU_TRI_BANG", "BAO_GÔM"]

    def close(self):
        self.driver.close()

    def _preprocess_question(self, question: str) -> list:
        question_lower = question.lower()
        question_clean = re.sub(r'[^\w\s]', '', question_lower)
        words = question_clean.split()
        
        stop_words = {"tôi", "bị", "là", "bệnh", "gì", "thì", "và", "uống", "thuốc", 
                      "để", "muốn", "tìm", "của", "có", "liên tục", "đang", "bị", "nào",
                      "cho", "của", "với", "tại", "đến", "từ", "về", "các", "mình", "ta",
                      "được", "phải", "này", "kia", "vậy", "nên", "rất", "quá", "lắm",
                      "thì", "mà", "là", "mà", "thì", "là", "bị", "đang", "liên tục"}
        
        # Lọc từ dừng
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        
        # Trích xuất cụm từ (bigram) - ưu tiên cụm từ có nghĩa
        bigrams = []
        for i in range(len(words) - 1):
            bigram = f"{words[i]} {words[i+1]}"
            # Chỉ giữ bigram không chứa từ dừng
            if not any(stop in bigram for stop in stop_words):
                bigrams.append(bigram)
        
        # Phân loại: cụm từ (dài > 3 ký tự) và từ đơn
        final_terms = []
        for term in bigrams:
            if len(term) > 5 and term not in final_terms:
                final_terms.append(term)
        for keyword in keywords:
            if keyword not in final_terms and len(keyword) > 2:
                final_terms.append(keyword)
        
        return final_terms

    def text_to_cypher(self, user_question: str) -> str:
        prompt = f"""
        You are an expert Neo4j Cypher developer for a Traditional Chinese Medicine database.
        Schema:
        {self.db_schema}
        
        CRITICAL RULES:
        1. EXTRACT FULL PHRASES: Extract full meaningful phrases (e.g., 'ho liên tục', 'khó thở').
        2. ALWAYS use `WHERE toLower(n.name) CONTAINS toLower('keyword')` for fuzzy searching.
        3. Keep Vietnamese accents strictly for relationships: CÓ_BIỂU_HIỆN, CHIA_THÀNH, ĐƯỢC_ĐIỀU_TRỊ_BẰNG, BAO_GÔM.
        4. Follow arrows strictly: (b:BenhLy)-[:CHIA_THÀNH]->(h:HoiChung)-[:CÓ_BIỂU_HIỆN]->(t:TrieuChung).
        5. DYNAMIC RETURN: 
           - ONLY diseases ("bệnh"): return `b.name`.
           - ONLY medicine ("thuốc"): return `p.name`.
           - BOTH or UNSPECIFIED intent (just listing symptoms): use `OPTIONAL MATCH` and return `b.name`, `p.name`.
        6. Output ONLY a SINGLE raw Cypher query. No explanations.
        7. FOR MULTIPLE SYMPTOMS (VERY IMPORTANT):
           - When the user mentions multiple symptoms (e.g., "ho nhiều, đờm trắng", "sốt cao, đau đầu"), you MUST use a separate `MATCH` clause for EACH symptom.
           - Each `MATCH` clause will connect the disease to a symptom through a syndrome.
           - Example for "ho nhiều đờm trắng":
             MATCH (b:BenhLy)-[:CHIA_THÀNH]->(h1:HoiChung)-[:CÓ_BIỂU_HIỆN]->(t1:TrieuChung)
             WHERE toLower(t1.name) CONTAINS toLower('ho nhiều')
             MATCH (b)-[:CHIA_THÀNH]->(h2:HoiChung)-[:CÓ_BIỂU_HIỆN]->(t2:TrieuChung)
             WHERE toLower(t2.name) CONTAINS toLower('đờm trắng')
             RETURN DISTINCT b.name
           - Do NOT use `AND` inside a single `WHERE` clause for multiple symptoms.
        
        --- EXAMPLES ---
        User: "Tôi bị đau đầu là bệnh gì?"
        Cypher: 
        MATCH (t:TrieuChung) WHERE toLower(t.name) CONTAINS toLower('đau đầu')
        MATCH (b:BenhLy)-[:CHIA_THÀNH]->(h:HoiChung)-[:CÓ_BIỂU_HIỆN]->(t)
        RETURN DISTINCT b.name
        
        User: "Tôi bị ho liên tục muốn tìm thuốc chữa"
        Cypher: 
        MATCH (t:TrieuChung) WHERE toLower(t.name) CONTAINS toLower('ho liên tục')
        MATCH (h:HoiChung)-[:CÓ_BIỂU_HIỆN]->(t)
        OPTIONAL MATCH (h)-[:ĐƯỢC_ĐIỀU_TRỊ_BẰNG]->(p:BaiThuoc)
        RETURN DISTINCT p.name
        
        User: "Tôi bị ho và khó thở"
        Cypher:
        MATCH (b:BenhLy)-[:CHIA_THÀNH]->(h1:HoiChung)-[:CÓ_BIỂU_HIỆN]->(t1:TrieuChung)
        WHERE toLower(t1.name) CONTAINS toLower('ho')
        MATCH (b)-[:CHIA_THÀNH]->(h2:HoiChung)-[:CÓ_BIỂU_HIỆN]->(t2:TrieuChung)
        WHERE toLower(t2.name) CONTAINS toLower('khó thở')
        OPTIONAL MATCH (h1)-[:ĐƯỢC_ĐIỀU_TRỊ_BẰNG]->(p:BaiThuoc)
        RETURN DISTINCT b.name, p.name
        ----------------

        Now translate this user question:
        User Question: "{user_question}"
        Cypher:
        """
        try:
            response = ollama.chat(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": "You are a precise Cypher query generator. Follow the instructions exactly."},
                    {"role": "user", "content": prompt}
                ],
                options={
                    "temperature": self.temperature,
                    "seed": self.seed,
                    "top_p": self.top_p
                }
            )
            cypher = response['message']['content'].strip()
            cypher = cypher.replace("```cypher", "").replace("```", "").strip()
            return cypher
        except Exception as e:
            logger.error(f"Lỗi sinh Cypher: {e}")
            return ""

    def run_cypher(self, cypher: str) -> list:
        """Thực thi Cypher trên Neo4j"""
        if not cypher:
            return []
        try:
            with self.driver.session() as session:
                result = session.run(cypher)
                return [dict(record) for record in result]
        except Exception as e:
            logger.error(f"Lỗi thực thi Cypher: {e}\nCypher: {cypher}")
            return []

    def _filter_results(self, records: list, terms: list) -> list:
        if not terms:
            return records
        
        # 1. Ưu tiên AND logic (tất cả triệu chứng phải có mặt)
        filtered_and = []
        for record in records:
            record_text = " ".join([str(val) for val in record.values()]).lower()
            if all(term in record_text for term in terms):
                filtered_and.append(record)
        
        # 2. Nếu có kết quả AND, trả về kết quả đó
        if filtered_and:
            return filtered_and
        
        # 3. Nếu không có AND, fallback sang OR
        logger.warning("Không tìm thấy AND, fallback sang OR")
        filtered_or = []
        for record in records:
            record_text = " ".join([str(val) for val in record.values()]).lower()
            if any(term in record_text for term in terms):
                filtered_or.append(record)
        
        return filtered_or

    def execute_and_answer(self, user_question: str) -> dict:
        """Chạy dịch câu hỏi, thực thi truy vấn Cypher và đóng gói câu trả lời tự nhiên"""
        # Bước 1: Tiền xử lý câu hỏi
        terms = self._preprocess_question(user_question)
        if not terms:
            return {
                "question": user_question,
                "answer": "Không thể trích xuất từ khóa từ câu hỏi của bạn.",
                "data": []
            }
        
        # Bước 2: Sinh Cypher
        cypher_query = self.text_to_cypher(user_question)
        if not cypher_query:
            return {
                "question": user_question,
                "cypher_used": "",
                "answer": "Không thể dịch câu hỏi sang câu truy vấn database.",
                "data": []
            }
        
        # Bước 3: Thực thi trong Neo4j
        try:
            records = self.run_cypher(cypher_query)
            if not records:
                return {
                    "question": user_question,
                    "cypher_used": cypher_query,
                    "answer": "Không tìm thấy kết quả phù hợp với truy vấn này.",
                    "data": []
                }
            
            # Bước 4: Lọc kết quả
            filtered_records = self._filter_results(records, terms)
            if not filtered_records:
                # Fallback: nếu không có kết quả sau lọc, dùng kết quả gốc
                logger.warning("Không tìm thấy kết quả khớp từ khóa, sử dụng kết quả gốc")
                filtered_records = records
            
            # Bước 5: Tự động đóng gói câu trả lời tự nhiên (NLG)
            danh_sach_benh = list(dict.fromkeys([record.get("b.name") for record in filtered_records if record.get("b.name")]))
            danh_sach_thuoc = list(dict.fromkeys([record.get("p.name") for record in filtered_records if record.get("p.name")]))
            danh_sach_hoi_chung = list(dict.fromkeys([record.get("h.name") for record in filtered_records if record.get("h.name")]))

            cau_tra_loi_parts = []
            
            if danh_sach_benh:
                gioi_han = danh_sach_benh[:5]
                text = f"Bệnh lý liên quan: {', '.join(gioi_han)}"
                if len(danh_sach_benh) > 5: text += f" (và {len(danh_sach_benh) - 5} bệnh khác)"
                cau_tra_loi_parts.append(text)
                
            if danh_sach_hoi_chung:
                gioi_han = danh_sach_hoi_chung[:5]
                text = f"Hội chứng: {', '.join(gioi_han)}"
                if len(danh_sach_hoi_chung) > 5: text += f" (và {len(danh_sach_hoi_chung) - 5} hội chứng khác)"
                cau_tra_loi_parts.append(text)

            if danh_sach_thuoc:
                gioi_han = danh_sach_thuoc[:5]
                text = f"Bài thuốc điều trị: {', '.join(gioi_han)}"
                if len(danh_sach_thuoc) > 5: text += f" (và {len(danh_sach_thuoc) - 5} bài khác)"
                cau_tra_loi_parts.append(text)

            # Gộp các câu trả lời lại
            if cau_tra_loi_parts:
                cau_tra_loi = "Dựa trên thông tin của bạn. " + " | ".join(cau_tra_loi_parts) + "."
            else:
                cau_tra_loi = "Hệ thống chưa tìm thấy thông tin phù hợp với truy vấn này."

            return {
                "question": user_question,
                "cypher_used": cypher_query,
                "answer": cau_tra_loi,
                "data": filtered_records
            }
            
        except Exception as e:
            logger.error(f"Lỗi khi chạy Cypher trên Neo4j: {e}")
            return {"error": str(e), "failed_cypher": cypher_query}

    def ask(self, question: str) -> dict:
        """Hỏi và nhận câu trả lời (tương thích CLI)"""
        return self.execute_and_answer(question)

    def _format_answer(self, data: list) -> str:
        """Chuyển kết quả thành văn bản tự nhiên"""
        if not data:
            return "Không tìm thấy kết quả."
        return json.dumps(data, ensure_ascii=False, indent=2)
