# src/qa_system.py
import json
import logging
import re
from neo4j import GraphDatabase
import ollama
from src.config_loader import load_config
from src.utils import normalize_symptoms_text

logger = logging.getLogger(__name__)

class TCMQA:
    def __init__(self, config: dict = None):
        self.config = config or load_config()
        self.llm_model = self.config.get("qa", {}).get("llm_model", "qwen2.5:7b")
        self.temperature = self.config.get("qa", {}).get("temperature", 0.0)
        self.seed = self.config.get("qa", {}).get("seed", 42)
        self.top_p = self.config.get("qa", {}).get("top_p", 0.9)

        # Cấu hình sử dụng SiliconFlow, OpenRouter, Hugging Face Cloud hoặc Ollama
        siliconflow_cfg = self.config.get("siliconflow", {})
        openrouter_cfg = self.config.get("openrouter", {})
        hf_cfg = self.config.get("huggingface", {})
        import os
        
        if siliconflow_cfg.get("use_cloud", False):
            token = os.environ.get("SILICONFLOW_API_KEY") or siliconflow_cfg.get("api_key")
            model_id = siliconflow_cfg.get("model", "Qwen/Qwen2.5-72B-Instruct")
            self.llm_model = model_id
            
            class SiliconFlowChatClient:
                def __init__(self, token_val: str, model_val: str, proxy: str = None):
                    self.token = token_val
                    self.model_id = model_val
                    self.url = "https://api.siliconflow.com/v1/chat/completions"
                    import httpx
                    headers = {
                        "Authorization": f"Bearer {self.token}",
                        "Content-Type": "application/json"
                    }
                    if proxy:
                        self.http_client = httpx.Client(proxies=proxy, headers=headers, timeout=120.0)
                        logger.info(f"Khởi tạo SiliconFlow Client cho model {model_val} qua proxy: {proxy}")
                    else:
                        self.http_client = httpx.Client(headers=headers, timeout=120.0)
                        logger.info(f"Khởi tạo SiliconFlow Client cho model: {model_val}")

                def chat(self, model: str, messages: list, options: dict = None) -> dict:
                    temperature = 0.0
                    if options:
                        if "temperature" in options:
                            temperature = options["temperature"]
                        if temperature == 0.0:
                            temperature = 0.01
                    payload = {
                        "model": self.model_id,
                        "messages": messages,
                        "temperature": temperature,
                        "stream": False
                    }
                    if options and "max_tokens" in options:
                        payload["max_tokens"] = options["max_tokens"]
                    try:
                        response = self.http_client.post(self.url, json=payload)
                        response.raise_for_status()
                        data = response.json()
                        content = data["choices"][0]["message"]["content"]
                        return {
                            "message": {
                                "role": "assistant",
                                "content": content
                            }
                        }
                    except Exception as e:
                        logger.error(f"Lỗi gọi SiliconFlow API: {e}")
                        if 'response' in locals() and response is not None:
                            logger.error(f"Chi tiết phản hồi lỗi: {response.text}")
                        raise e

            proxy = siliconflow_cfg.get("proxy")
            self.client = SiliconFlowChatClient(token, model_id, proxy)
            logger.info("TCMQA kết nối SiliconFlow thành công!")
            
        elif openrouter_cfg.get("use_cloud", False):
            token = os.environ.get("OPENROUTER_API_KEY") or openrouter_cfg.get("api_key")
            model_id = openrouter_cfg.get("model", "qwen/qwen-2.5-vl-72b-instruct:free")
            self.llm_model = model_id
            
            class OpenRouterChatClient:
                def __init__(self, token_val: str, model_val: str, proxy: str = None):
                    self.token = token_val
                    self.model_id = model_val
                    self.url = "https://openrouter.ai/api/v1/chat/completions"
                    import httpx
                    headers = {
                        "Authorization": f"Bearer {self.token}",
                        "Content-Type": "application/json",
                        "HTTP-Referer": "http://localhost:3000",
                        "X-Title": "TCM-Qwen-QA"
                    }
                    if proxy:
                        self.http_client = httpx.Client(proxies=proxy, headers=headers, timeout=60.0)
                        logger.info(f"Khởi tạo OpenRouter Client cho model {model_val} qua proxy: {proxy}")
                    else:
                        self.http_client = httpx.Client(headers=headers, timeout=60.0)
                        logger.info(f"Khởi tạo OpenRouter Client cho model: {model_val}")

                def chat(self, model: str, messages: list, options: dict = None) -> dict:
                    temperature = 0.0
                    if options:
                        if "temperature" in options:
                            temperature = options["temperature"]
                        if temperature == 0.0:
                            temperature = 0.01
                    payload = {
                        "model": self.model_id,
                        "messages": messages,
                        "temperature": temperature,
                        "stream": False
                    }
                    if options and "max_tokens" in options:
                        payload["max_tokens"] = options["max_tokens"]
                    try:
                        response = self.http_client.post(self.url, json=payload)
                        response.raise_for_status()
                        data = response.json()
                        content = data["choices"][0]["message"]["content"]
                        return {
                            "message": {
                                "role": "assistant",
                                "content": content
                            }
                        }
                    except Exception as e:
                        logger.error(f"Lỗi gọi OpenRouter API: {e}")
                        if 'response' in locals() and response is not None:
                            logger.error(f"Chi tiết phản hồi lỗi: {response.text}")
                        raise e

            proxy = openrouter_cfg.get("proxy")
            self.client = OpenRouterChatClient(token, model_id, proxy)
            logger.info("TCMQA kết nối OpenRouter thành công!")
            
        elif hf_cfg.get("use_cloud", False):
            # Khởi tạo client Hugging Face Serverless
            token = os.environ.get("HF_TOKEN") or hf_cfg.get("token")
            model_id = hf_cfg.get("model", "Qwen/Qwen2.5-72B-Instruct")
            self.llm_model = model_id
            
            class HuggingFaceChatClient:
                def __init__(self, token_val: str, model_val: str, proxy: str = None):
                    self.token = token_val
                    self.model_id = model_val
                    self.url = "https://router.huggingface.co/v1/chat/completions"
                    import httpx
                    if proxy:
                        self.http_client = httpx.Client(proxies=proxy, timeout=60.0)
                        logger.info(f"Khởi tạo HuggingFace Serverless Client cho model {model_val} qua proxy: {proxy}")
                    else:
                        self.http_client = httpx.Client(timeout=60.0)
                        logger.info(f"Khởi tạo HuggingFace Serverless Client cho model: {model_val}")

                def chat(self, model: str, messages: list, options: dict = None) -> dict:
                    headers = {
                        "Authorization": f"Bearer {self.token}",
                        "Content-Type": "application/json"
                    }
                    temperature = 0.0
                    if options:
                        if "temperature" in options:
                            temperature = options["temperature"]
                        if temperature == 0.0:
                            temperature = 0.01
                    payload = {
                        "model": self.model_id,
                        "messages": messages,
                        "temperature": temperature,
                        "stream": False
                    }
                    if options and "max_tokens" in options:
                        payload["max_tokens"] = options["max_tokens"]
                    try:
                        response = self.http_client.post(self.url, headers=headers, json=payload)
                        response.raise_for_status()
                        data = response.json()
                        content = data["choices"][0]["message"]["content"]
                        return {
                            "message": {
                                "role": "assistant",
                                "content": content
                            }
                        }
                    except Exception as e:
                        logger.error(f"Lỗi gọi Hugging Face Serverless API: {e}")
                        if 'response' in locals() and response is not None:
                            logger.error(f"Chi tiết phản hồi lỗi: {response.text}")
                        raise e

            proxy = hf_cfg.get("proxy")
            self.client = HuggingFaceChatClient(token, model_id, proxy)
            logger.info("TCMQA kết nối Hugging Face Cloud Inference API thành công!")
        else:
            # Hỗ trợ host remote
            self.ollama_host = self.config.get("host") or self.config.get("ollama", {}).get("host")
            if self.ollama_host:
                from ollama import Client
                self.client = Client(host=self.ollama_host)
                logger.info(f"TCMQA kết nối Ollama remote host: {self.ollama_host}")
            else:
                import ollama
                self.client = ollama
                logger.info("TCMQA kết nối Ollama local host")

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
        self.rel_types = ["CHIA_THÀNH", "CÓ_BIỂU_HIỆN", "ĐƯỢC_ĐIỀU_TRỊ_BẰNG", "BAO_GỒM"]

    def close(self):
        self.driver.close()

    def _extract_symptoms_by_matching(self, text: str) -> list:
        if not text:
            return []
        text_lower = text.lower()
        
        # Lấy tất cả triệu chứng từ database
        try:
            db_symptoms = self.neo4j_client.get_all_symptoms()
        except Exception as e:
            try:
                # Nếu neo4j_client.get_all_symptoms() không có/chưa khởi tạo, query trực tiếp từ self.driver
                with self.driver.session() as session:
                    result = session.run("MATCH (t:TrieuChung) RETURN t.name AS name")
                    db_symptoms = [record["name"] for record in result]
            except Exception as ex:
                logger.error(f"Lỗi truy vấn tất cả triệu chứng: {ex}")
                db_symptoms = []
            
        # Sắp xếp theo chiều dài giảm dần để ưu tiên triệu chứng dài trước
        db_symptoms_sorted = sorted(db_symptoms, key=len, reverse=True)
        
        extracted = []
        temp_text = text_lower
        
        for symptom in db_symptoms_sorted:
            sym_l = symptom.lower()
            toks = sym_l.split()
            if len(toks) < 2:
                continue
            # Dùng regex \b để khớp từ độc lập tránh substring trượt.
            # Với triệu chứng ĐÚNG 2 TỪ: chấp nhận ĐẢO THỨ TỰ (vd DB 'họng ngứa' khớp cả
            # khi người dùng gõ 'ngứa họng') để vấn chẩn không phụ thuộc thứ tự từ.
            if len(toks) == 2:
                a, b = re.escape(toks[0]), re.escape(toks[1])
                pattern = rf'\b(?:{a}\s+{b}|{b}\s+{a})\b'
            else:
                pattern = rf'\b{re.escape(sym_l)}\b'
            if re.search(pattern, temp_text):
                extracted.append(symptom)
                # Thay bằng khoảng trắng để tránh các cụm từ khác đè trùng
                temp_text = re.sub(pattern, " " * len(sym_l), temp_text, count=1)

        return extracted

    def _preprocess_question(self, question: str) -> list:
        # Chuẩn hóa văn bản trước khi khớp
        normalized_q = normalize_symptoms_text(question)
        # Sử dụng thuật toán khớp triệu chứng trực tiếp từ database (Longest Match First)
        return self._extract_symptoms_by_matching(normalized_q)

    def text_to_cypher(self, user_question: str, terms: list = None) -> str:
        if terms is None:
            terms = self._preprocess_question(user_question)
        if not terms:
            return ""
            
        symptoms_str = ", ".join([f"'{t}'" for t in terms])
        
        prompt = f"""
        You are an expert Neo4j Cypher developer for a Traditional Chinese Medicine database.
        Schema:
        {self.db_schema}
        
        CRITICAL RULES FOR CYPHER GENERATION:
        1. USE THE PROVIDED SYMPTOMS ONLY: Generate the Cypher query using exactly the following medical symptoms extracted from the user's question: {symptoms_str}.
           Do NOT extract, change, or add any other symptoms.
        
        2. EXACT WORD MATCHING (CRITICAL): NEVER use CONTAINS to prevent substring bugs. You MUST use Unicode-aware word boundaries `(^|[^\\p{{L}}])keyword($|[^\\p{{L}}])`.
            Syntax: WHERE toLower(t.name) =~ '.*(^|[^\\p{{L}}])keyword($|[^\\p{{L}}]).*' (keyword must be in lowercase, e.g., 'ho' instead of 'Ho')
            Example: For symptom 'ho liên tục', use: WHERE toLower(t.name) =~ '.*(^|[^\\p{{L}}])ho liên tục($|[^\\p{{L}}]).*'
         
                  3. MANDATORY RETURN STRUCTURE (NO EXCEPTIONS):
            MATCH (b:BenhLy)-[:CHIA_THÀNH]->(h:HoiChung)-[r:CÓ_BIỂU_HIỆN]->(t:TrieuChung)
            WHERE toLower(t.name) =~ '.*(^|[^\\p{{L}}])keyword1($|[^\\p{{L}}]).*'
              AND r.benh_ly = b.name
            OPTIONAL MATCH (h)-[:ĐƯỢC_ĐIỀU_TRỊ_BẰNG]->(p:BaiThuoc)
            WHERE p.benh_ly = b.name AND p.hoi_chung = h.name
            RETURN DISTINCT b.name, h.name, p.name
         
                  4. FOR MULTIPLE SYMPTOMS: Use multiple MATCH clauses.
            Example for symptoms 'ho khan' and 'đau đầu':
            MATCH (b:BenhLy)-[:CHIA_THÀNH]->(h:HoiChung)
            MATCH (h)-[r1:CÓ_BIỂU_HIỆN]->(t1:TrieuChung) WHERE toLower(t1.name) =~ '.*(^|[^\\p{{L}}])ho khan($|[^\\p{{L}}]).*' AND r1.benh_ly = b.name
            MATCH (h)-[r2:CÓ_BIỂU_HIỆN]->(t2:TrieuChung) WHERE toLower(t2.name) =~ '.*(^|[^\\p{{L}}])đau đầu($|[^\\p{{L}}]).*' AND r2.benh_ly = b.name
            OPTIONAL MATCH (h)-[:ĐƯỢC_ĐIỀU_TRỊ_BẰNG]->(p:BaiThuoc)
            WHERE p.benh_ly = b.name AND p.hoi_chung = h.name
            RETURN DISTINCT b.name, h.name, p.name
        

        5. Output ONLY a SINGLE raw Cypher query. Do NOT use markdown code blocks (like ```cypher). No explanations.
        
        User Question: "{user_question}"
        Cypher:
        """
        try:
            response = self.client.chat(
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
            
            # Sửa các lỗi chính tả/dấu cách trong tên mối quan hệ do LLM sinh ra
            cypher = cypher.replace("CÓ BIỂU HIỆN", "CÓ_BIỂU_HIỆN")
            cypher = cypher.replace("CÓ_BIỂU HIỆN", "CÓ_BIỂU_HIỆN")
            cypher = cypher.replace("CÓ BIỂU_HIỆN", "CÓ_BIỂU_HIỆN")
            cypher = cypher.replace("CHIA THÀNH", "CHIA_THÀNH")
            cypher = cypher.replace("ĐƯỢC ĐIỀU TRỊ BẰNG", "ĐƯỢC_ĐIỀU_TRỊ_BẰNG")
            cypher = cypher.replace("ĐƯỢC_ĐIỀU_TRỊ BẰNG", "ĐƯỢC_ĐIỀU_TRỊ_BẰNG")
            cypher = cypher.replace("BAO GỒM", "BAO_GỒM")
            cypher = cypher.replace("BAO GÔM", "BAO_GỒM")
            cypher = cypher.replace("BAO_GÔM", "BAO_GỒM")
            
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

    # ====================================================================
    # [FIX] TRUY HỒI NHẤT QUÁN THEO ĐỒ THỊ (giữ đúng ngữ cảnh Bệnh–Hội chứng–Bài thuốc)
    # Khắc phục 2 lỗi của pipeline web:
    #   - "Sai bài thuốc": trước đây lấy LIMIT 1 bài thuốc bất kỳ của hội chứng dùng chung.
    #   - "Thiếu kết quả": trước đây khớp '=' chính xác nên bỏ sót triệu chứng nằm trong node ghép.
    # ====================================================================
    @staticmethod
    def _word_boundary_pattern(term: str) -> str:
        """Tạo Java-regex khớp 'term' như một cụm từ độc lập (tránh dính chuỗi con),
        bắt được cả khi term nằm trong node triệu chứng ghép.
        Với cụm ĐÚNG 2 TỪ: cho phép ĐẢO THỨ TỰ (vd 'ngứa họng' khớp cả node 'họng ngứa')
        để vấn chẩn không phụ thuộc thứ tự từ người dùng gõ."""
        # \Q...\E: trích dẫn nguyên văn, an toàn với ký tự đặc biệt như dấu ngoặc.
        t = term.lower().strip()
        toks = t.split()
        if len(toks) == 2:
            a, b = toks
            core = r'(\Q' + a + r'\E \Q' + b + r'\E|\Q' + b + r'\E \Q' + a + r'\E)'
        else:
            core = r'\Q' + t + r'\E'
        return r'(?s).*(^|[^\p{L}])' + core + r'($|[^\p{L}]).*'

    def get_symptom_disease_map(self, terms: list) -> dict:
        """Trả về {hội chứng: [danh sách bệnh]} mà triệu chứng người bệnh THỰC SỰ thuộc về.
        Đi đúng đường đồ thị và ràng buộc r.benh_ly = b.name để loại các bệnh không liên quan."""
        mapping = {}
        for term in (terms or []):
            if not term:
                continue
            pattern = self._word_boundary_pattern(term)
            # CHẶT: bắt buộc r.benh_ly = b.name để cột chặt triệu chứng vào đúng bệnh của nó.
            # (DB có ~49% cạnh benh_ly=NULL trùng lặp; nếu nới 'IS NULL OR' sẽ khớp tràn sang mọi
            # bệnh dùng chung hội chứng. Đã kiểm chứng: 0 cặp (HC,triệu chứng) chỉ có bản NULL,
            # nên ràng buộc chặt KHÔNG bỏ sót kết quả nào.)
            cypher = """
            MATCH (b:BenhLy)-[:CHIA_THÀNH]->(h:HoiChung)-[r:CÓ_BIỂU_HIỆN]->(t:TrieuChung)
            WHERE toLower(t.name) =~ $pattern AND r.benh_ly = b.name
            RETURN DISTINCT h.name AS syndrome, b.name AS disease
            """
            try:
                with self.driver.session() as session:
                    for rec in session.run(cypher, pattern=pattern):
                        if rec["syndrome"] and rec["disease"]:
                            mapping.setdefault(rec["syndrome"], set()).add(rec["disease"])
            except Exception as e:
                logger.error(f"Lỗi get_symptom_disease_map (term='{term}'): {e}")
        return {k: sorted(v) for k, v in mapping.items()}

    def get_treatments_for_syndrome(self, syndrome: str, diseases: list = None) -> list:
        """Trả về [{disease, bai_thuoc, vi_thuoc}] NHẤT QUÁN: mỗi bệnh đi kèm ĐÚNG bài thuốc của nó
        (ràng buộc p.benh_ly = b.name AND p.hoi_chung = h.name). Nếu diseases=None thì lấy mọi bệnh."""
        cypher = """
        MATCH (b:BenhLy)-[:CHIA_THÀNH]->(h:HoiChung {name: $syn})
        WHERE $diseases IS NULL OR b.name IN $diseases
        OPTIONAL MATCH (h)-[:ĐƯỢC_ĐIỀU_TRỊ_BẰNG]->(p:BaiThuoc)
          WHERE p.benh_ly = b.name AND p.hoi_chung = h.name
        OPTIONAL MATCH (p)-[:BAO_GỒM]->(v:ViThuoc)
        RETURN b.name AS disease, p.name AS bai_thuoc, collect(DISTINCT v.name) AS vi_thuoc
        ORDER BY disease
        """
        out = []
        try:
            with self.driver.session() as session:
                for rec in session.run(cypher, syn=syndrome, diseases=diseases):
                    out.append({
                        "disease": rec["disease"],
                        "bai_thuoc": rec["bai_thuoc"],
                        "vi_thuoc": [v for v in (rec["vi_thuoc"] or []) if v],
                    })
        except Exception as e:
            logger.error(f"Lỗi get_treatments_for_syndrome ('{syndrome}'): {e}")
        return out

    def _filter_results(self, records: list, terms: list) -> list:
        if not terms:
            return records
        
        import re # Đảm bảo đã import re ở đầu file
        
        # 1. Ưu tiên AND logic
        filtered_and = []
        for record in records:
            record_text = " ".join([str(val) for val in record.values()]).lower()
            # Dùng regex \b để bắt buộc từ khóa phải đứng độc lập (vd: \bho\b không match choáng)
            if all(re.search(rf'\b{re.escape(term.lower())}\b', record_text) for term in terms):
                filtered_and.append(record)
        
        if filtered_and:
            return filtered_and
        
        # 2. Fallback sang OR
        logger.warning("Không tìm thấy AND, fallback sang OR")
        filtered_or = []
        for record in records:
            record_text = " ".join([str(val) for val in record.values()]).lower()
            if any(re.search(rf'\b{re.escape(term.lower())}\b', record_text) for term in terms):
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
        cypher_query = self.text_to_cypher(user_question, terms=terms)
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
