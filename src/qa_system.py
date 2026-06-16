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
            neo4j_cfg.get("uri", "neo4j+s://c55f875f.databases.neo4j.io"),
            auth=(neo4j_cfg.get("user", "c55f875f"), neo4j_cfg.get("password", "Z7b-auwCd7T1KPY8TF0p3_piWcAyfospK55nC196c7w"))
        )
        logger.info(f"Đã kết nối Neo4j: {neo4j_cfg.get('uri')}")

        # Đọc schema
        with open("data/graph_schema.txt", "r", encoding="utf-8") as f:
            self.schema = f.read()
        self.db_schema = self.schema
        logger.info(f"Đã tải schema từ data/graph_schema.txt")

        # Danh sách nhãn và quan hệ để LLM biết
        self.node_labels = ["BenhLy", "HoiChung", "TrieuChung", "BaiThuoc", "ViThuoc"]
        self.rel_types = ["CHIA_THANH", "CO_BIEU_HIEN", "DUOC_DIEU_TRI_BANG", "BAO_GOM"]

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
        
        CRITICAL RULES FOR CYPHER GENERATION:
        1. CORE MEDICAL KEYWORDS ONLY: Extract ONLY the core medical symptoms. Ignore words like tôi, bị, liên tục...
        
        2. EXACT WORD MATCHING (CRITICAL): NEVER use CONTAINS to prevent substring bugs (where 'ho' falsely matches 'choáng'). You MUST use Regex word boundaries `\\\\b`.
           Syntax: WHERE toLower(t.name) =~ '.*\\\\bkeyword\\\\b.*'
           Example: WHERE toLower(t.name) =~ '.*\\\\bho\\\\b.*'
        
        3. MANDATORY RETURN STRUCTURE (NO EXCEPTIONS):
           MATCH (h:HoiChung)-[:CO_BIEU_HIEN]->(t:TrieuChung)
           WHERE toLower(t.name) =~ '.*\\\\bkeyword1\\\\b.*'
           OPTIONAL MATCH (b:BenhLy)-[:CHIA_THANH]->(h)
           OPTIONAL MATCH (h)-[:DUOC_DIEU_TRI_BANG]->(p:BaiThuoc)
           RETURN DISTINCT b.name, h.name, p.name
        
        4. FOR MULTIPLE SYMPTOMS: Use multiple MATCH clauses.
           Example:
           MATCH (h:HoiChung)-[:CO_BIEU_HIEN]->(t1:TrieuChung) WHERE toLower(t1.name) =~ '.*\\\\bho\\\\b.*'
           MATCH (h)-[:CO_BIEU_HIEN]->(t2:TrieuChung) WHERE toLower(t2.name) =~ '.*\\\\bsốt\\\\b.*'
           OPTIONAL MATCH (b:BenhLy)-[:CHIA_THANH]->(h)
           OPTIONAL MATCH (h)-[:DUOC_DIEU_TRI_BANG]->(p:BaiThuoc)
           RETURN DISTINCT b.name, h.name, p.name

        5. Output ONLY a SINGLE raw Cypher query. Do NOT use markdown code blocks (like ```cypher). No explanations.
        
        User Question: "{user_question}"
        Cypher:
        """
        try:
            import ollama
            response = ollama.chat(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": "You are a precise Cypher query generator. Output only the raw query without any markdown formatting."},
                    {"role": "user", "content": prompt}
                ],
                options={
                    "temperature": 0.0,  # Đưa temperature về 0.0 để loại bỏ hoàn toàn tính ngẫu nhiên
                    "seed": self.seed,
                    "top_p": self.top_p
                }
            )
            cypher = response['message']['content'].strip()
            # Dọn dẹp ký tự markdown nếu LLM lỡ sinh ra
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
        
        import re # Đảm bảo đã import re ở đầu file
        
        # 1. Ưu tiên AND logic
        filtered_and = []
        for record in records:
            record_text = " ".join([str(val) for val in record.values()]).lower()
            # Dùng regex \b để bắt buộc từ khóa phải đứng độc lập (vd: \bho\b không match choáng)
            if all(re.search(rf'\b{re.escape(term)}\b', record_text) for term in terms):
                filtered_and.append(record)
        
        if filtered_and:
            return filtered_and
        
        # 2. Fallback sang OR
        logger.warning("Không tìm thấy AND, fallback sang OR")
        filtered_or = []
        for record in records:
            record_text = " ".join([str(val) for val in record.values()]).lower()
            if any(re.search(rf'\b{re.escape(term)}\b', record_text) for term in terms):
                filtered_or.append(record)
        
        return filtered_or

    def execute_and_answer(self, user_question: str) -> dict:
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
            
            # Bước 4: Lọc kết quả bằng logic ưu tiên
            # Nhóm các bệnh theo tên
            disease_map = {}
            for record in records:
                disease = record.get("b.name", "")
                if not disease:
                    continue
                if disease not in disease_map:
                    disease_map[disease] = []
                disease_map[disease].append(record)
            
            # Tính điểm ưu tiên cho mỗi bệnh dựa trên số lượng hội chứng và bài thuốc
            scored_diseases = []
            for disease, records_list in disease_map.items():
                # Đếm số lượng hội chứng và bài thuốc duy nhất
                syndromes = set()
                prescriptions = set()
                for record in records_list:
                    if record.get("h.name"):
                        syndromes.add(record.get("h.name"))
                    if record.get("p.name"):
                        prescriptions.add(record.get("p.name"))
                
                score = len(syndromes) + len(prescriptions)
                scored_diseases.append({
                    "disease": disease,
                    "score": score,
                    "records": records_list,
                    "syndromes": list(syndromes),
                    "prescriptions": list(prescriptions)
                })
            
            # Sắp xếp theo điểm giảm dần
            scored_diseases.sort(key=lambda x: x["score"], reverse=True)
            
            # Chỉ lấy top 10 kết quả tốt nhất
            top_diseases = scored_diseases[:10]
            
            # Bước 5: Đóng gói câu trả lời tự nhiên (NLG)
            cau_tra_loi_parts = []
            
            if top_diseases:
                danh_sach_benh = [d["disease"] for d in top_diseases]
                gioi_han = danh_sach_benh[:5]
                text = f"Bệnh lý liên quan (ưu tiên theo độ khớp): {', '.join(gioi_han)}"
                if len(danh_sach_benh) > 5:
                    text += f" (và {len(danh_sach_benh) - 5} bệnh khác)"
                cau_tra_loi_parts.append(text)
                
                # Lấy bài thuốc cho bệnh có điểm cao nhất
                best_disease = top_diseases[0]
                if best_disease.get("prescriptions"):
                    text = f"Bài thuốc ưu tiên cho bệnh '{best_disease['disease']}': {', '.join(best_disease['prescriptions'])}"
                    cau_tra_loi_parts.append(text)
            
            if cau_tra_loi_parts:
                cau_tra_loi = "Dựa trên thông tin của bạn. " + " | ".join(cau_tra_loi_parts) + "."
            else:
                cau_tra_loi = "Hệ thống chưa tìm thấy thông tin phù hợp với truy vấn này."
            
            return {
                "question": user_question,
                "cypher_used": cypher_query,
                "answer": cau_tra_loi,
                "data": records
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
