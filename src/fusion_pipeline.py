# src/fusion_pipeline.py
import logging
import re
from src.pipeline import TCMTonguePipeline
from src.qa_system import TCMQA

from src.utils import normalize_symptoms_text

logger = logging.getLogger(__name__)

class TCMFusionPipeline:
    def __init__(self, config: dict = None):
        """Khởi tạo toàn bộ lõi AI của hệ thống"""
        logger.info("Đang khởi tạo Hệ thống Hợp nhất (Fusion Pipeline)...")
        from src.config_loader import load_config
        config = config or load_config()

        # Module Vọng chẩn (LLaVA - Phân tích ảnh)
        self.vision_pipeline = TCMTonguePipeline(config=config)

        # Module Vấn chẩn & Đồ thị (Qwen + Neo4j)
        self.qa_pipeline = TCMQA(config=config)

        self.qa_pipeline.neo4j_client = self.vision_pipeline.neo4j_client

        # Tải danh sách các hội chứng hợp lệ từ database để tránh ảo giác
        try:
            self.valid_syndromes = self.vision_pipeline.neo4j_client.get_all_syndromes()
            logger.info(f"Đã tải {len(self.valid_syndromes)} hội chứng hợp lệ từ Neo4j")
        except Exception as e:
            logger.error(f"Lỗi khi tải danh sách hội chứng từ database: {e}")
            self.valid_syndromes = []

        # Tải danh sách triệu chứng lưỡi và mặt chuẩn từ file mapping để mapping chính xác
        import json
        import os
        self.tongue_symptoms_list = []
        self.face_symptoms_list = []
        
        tongue_map_path = "data/mapping/symptom_to_syndrome.json"
        if os.path.exists(tongue_map_path):
            try:
                with open(tongue_map_path, "r", encoding="utf-8") as f:
                    self.tongue_symptoms_list = list(json.load(f).keys())
                logger.info(f"Đã tải {len(self.tongue_symptoms_list)} triệu chứng lưỡi chuẩn từ mapping")
            except Exception as e:
                logger.error(f"Lỗi tải triệu chứng lưỡi từ mapping: {e}")
                
        face_map_path = "data/mapping/face_to_syndrome.json"
        if os.path.exists(face_map_path):
            try:
                with open(face_map_path, "r", encoding="utf-8") as f:
                    self.face_symptoms_list = list(json.load(f).keys())
                logger.info(f"Đã tải {len(self.face_symptoms_list)} triệu chứng mặt chuẩn từ mapping")
            except Exception as e:
                logger.error(f"Lỗi tải triệu chứng mặt từ mapping: {e}")

        if not self.tongue_symptoms_list:
            self.tongue_symptoms_list = [
                "Lưỡi bệu có dấu răng", "Rêu lưỡi trắng mỏng", "Lưỡi đỏ",
                "Rêu vàng dày", "Lưỡi nhợt", "Rêu bong tróc", "Lưỡi tím",
                "Lưỡi có vết nứt", "Lưỡi sưng", "Lưỡi khô", "Rêu lưỡi vàng nhầy",
                "Lưỡi nhỏ đỏ", "Lưỡi có dấu răng", "Lưỡi không có rêu", "Loét miệng lưỡi sưng đau",
                "Đầu lưỡi đỏ", "Loét lưỡi", "Lưỡi nứt"
            ]
        if not self.face_symptoms_list:
            self.face_symptoms_list = [
                "Mặt đỏ", "Mặt trắng nhợt", "Mặt nhợt nhạt", "Mặt vàng", "Mặt xanh",
                "Mặt đen", "Mặt phù", "Mặt xám ngoét", "Sắc mặt ám tối", "Mặt có ban",
                "Ban xuất huyết dưới da", "Da xuất hiện ban đỏ hình bướm", "Da có mảng đỏ có vảy",
                "Da nổi mụn nước", "Da ngứa", "Da khô", "Mắt đỏ", "Mắt lồi", "Môi méo", "Môi thâm"
            ]

        logger.info("Khởi tạo hoàn tất!")

    def _clean_foreign_characters(self, text: str) -> str:
        """Loại bỏ toàn bộ chữ Hán, chữ Cyrillic (Nga), token yka3 và ký tự rác để đảm bảo đầu ra sạch 100%"""
        if not text:
            return ""
        # Xóa yka3 (case-insensitive)
        text = re.sub(r'(?i)\byka3\b', '', text)
        text = text.replace("yka3", "").replace("Yka3", "")
        # Xóa toàn bộ chữ Hán và chữ Cyrillic
        text = re.sub(r'[\u4e00-\u9fff\u0400-\u04ff]', '', text)
        # Xóa các ký tự dấu câu Trung Quốc đặc thù
        chinese_symbols = ['：', '，', '。', '！', '？', '（', '）', '【', '】', '“', '”', '‘', '’', '；']
        for sym in chinese_symbols:
            text = text.replace(sym, '')
        # Định dạng lại khoảng trắng thừa
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n\s*\n+', '\n\n', text)
        return text.strip()

    def _clean_translated_text(self, text: str) -> str:
        """Tách khối dịch bị lặp/glitch của Qwen và trả về phần dịch chuẩn tiếng Việt cuối cùng"""
        if not text:
            return ""
            
        # 1. Xóa các từ lặp lại liên tục do lỗi lặp từ của Qwen (ví dụ: abnormalities-abnormalities-...)
        # \b(\w+)\b: bắt từ
        # (?:[\s\-_]+\b\1\b){2,}: tìm từ đó xuất hiện tiếp theo từ 2 lần trở lên
        text = re.sub(r'\b(\w+)\b(?:[\s\-_]+\b\1\b){2,}', r'\1', text, flags=re.IGNORECASE)
            
        pattern = r'[\u4e00-\u9fff\u0400-\u04ff]|[yY][kK][aA]3'
        
        # Nếu phát hiện lỗi tokenizer hoặc chữ Hán/Nga, tiến hành tách đoạn để lấy bản dịch sạch ở cuối
        if re.search(pattern, text):
            # Tách bằng chuỗi ký tự lỗi/chữ Hán/chữ Nga hoặc dấu hai chấm
            split_pattern = r'(?:[\u4e00-\u9fff\u0400-\u04ff]|[yY][kK][aA]3|：|，|。|！|？|（|）|【|】)+'
            parts = re.split(split_pattern, text)
            parts = [p.strip() for p in parts if p.strip()]
            
            if parts:
                last_part = parts[-1]
                if len(last_part) > 20:
                    text = last_part
                else:
                    text = max(parts, key=len)
                    
        # 1.5. Dịch các từ tiếng Anh chuyên ngành thường gặp mà Qwen đôi khi bỏ sót
        replacements = {
            r'\bExpression\b': 'Biểu cảm',
            r'\bexpression\b': 'biểu cảm',
            r'\bComplexion\b': 'Sắc mặt',
            r'\bcomplexion\b': 'sắc mặt',
            r'\bAbnormalities\b': 'Đặc điểm bất thường',
            r'\babnormalities\b': 'đặc điểm bất thường',
            r'\bAbnormality\b': 'Đặc điểm bất thường',
            r'\babnormality\b': 'đặc điểm bất thường',
            r'\bTongue\b': 'Lưỡi',
            r'\btongue\b': 'lưỡi',
            r'\bFace\b': 'Mặt',
            r'\bface\b': 'mặt'
        }
        for eng, vie in replacements.items():
            text = re.sub(eng, vie, text)
            
        # Áp dụng bộ lọc ký tự ngoại quốc chung
        text = self._clean_foreign_characters(text)
        
        # Chuẩn hóa dấu chấm câu
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\.+', '.', text)
        text = text.strip()
        if text and not text.endswith('.'):
            text += '.'
        return text

    def _translate_english_description(self, english_text: str) -> str:
        """Dịch mô tả sắc mặt/lưỡi tiếng Anh sang tiếng Việt bằng Qwen"""
        if not english_text:
            return ""
        
        # Heuristic: Thay thế các từ nhạy cảm để tránh bug tokenization/loop của Qwen
        text_clean = english_text.replace("indicate", "suggest").replace("Indicate", "Suggest")
        text_clean = text_clean.replace("indicates", "suggests").replace("Indicates", "Suggests")
        text_clean = text_clean.replace("abnormalities", "abnormal features").replace("Abnormalities", "Abnormal features")
        text_clean = text_clean.replace("abnormality", "abnormal feature").replace("Abnormality", "Abnormal feature")
        
        prompt = f"""
        Translate the following English medical description of patient's features into natural Vietnamese.
        Output ONLY the translated Vietnamese text. Do not add any explanation or metadata.

        Text to translate: "{text_clean}"
        Vietnamese translation:
        """
        try:
            res = self.qa_pipeline.client.chat(
                model=self.qa_pipeline.llm_model,
                messages=[
                    {"role": "system", "content": "Bạn là dịch giả y khoa Đông y chuyên nghiệp. Bạn chỉ dịch văn bản tiếng Anh sang tiếng Việt. Bạn chỉ được viết bằng chữ cái tiếng Việt, tuyệt đối không dùng chữ Hán, chữ Trung Quốc hay ký tự ngoại quốc nào khác."},
                    {"role": "user", "content": prompt}
                ],
                options={"temperature": 0.0, "seed": 42, "max_tokens": 150}
            )
            ans = res['message']['content'].strip()
            ans = ans.replace("Vietnamese translation:", "").replace("Vietnamese:", "").replace('"', '').strip()
            
            # Làm sạch kết quả dịch triệt để chống loop và chữ ngoại quốc
            ans = self._clean_translated_text(ans)
            return ans
        except Exception as e:
            logger.error(f"Lỗi dịch mô tả tiếng Anh: {e}")
            return self._clean_translated_text(english_text)

    def _map_desc_to_symptoms(self, description: str, candidate_symptoms: list) -> list:
        """Dùng Qwen để ánh xạ mô tả tự nhiên tiếng Anh/tiếng Việt sang các triệu chứng chuẩn trong database"""
        if not description:
            return []
            
        candidates_str = ", ".join(candidate_symptoms)
        
        prompt = f"""
        Role: Traditional Chinese Medicine expert assistant.
        We have a list of standardized medical symptoms:
        {candidates_str}
        
        Analyze the following patient feature description:
        "{description}"
        
        Task: Select all symptoms from the standardized list above that are mentioned or described in the patient description.
        Only select symptoms that are actually present.
        
        CRITICAL RULES:
        1. Output ONLY a comma-separated list of the selected standardized symptoms, EXACTLY as written in the list.
        2. Do NOT add any introductory text, explanation, notes, or markdown.
        3. If no symptoms match, output 'Không'.
        
        Standardized symptoms:
        """
        try:
            res = self.qa_pipeline.client.chat(
                model=self.qa_pipeline.llm_model,
                messages=[
                    {"role": "system", "content": "You are a precise data mapper. Output only the comma-separated symptoms from the list, or 'Không'."},
                    {"role": "user", "content": prompt}
                ],
                options={"temperature": 0.0, "seed": 42}
            )
            content = res['message']['content'].strip()
            if content.lower() == 'không' or not content:
                return []
            
            # Split and clean the mapped symptoms
            mapped = [s.strip() for s in content.split(",") if s.strip()]
            # Filter to make sure only exact matches are returned (case-insensitive check but keep exact spelling)
            valid_mapped = []
            for m in mapped:
                for cand in candidate_symptoms:
                    if m.lower() == cand.lower() and cand not in valid_mapped:
                        valid_mapped.append(cand)
            return valid_mapped
        except Exception as e:
            logger.error(f"Lỗi mapping triệu chứng bằng LLM: {e}")
            return []

    def _extract_syndromes_from_text(self, text: str) -> list:
        """Trích xuất hội chứng bằng cách khớp triệu chứng chính xác trước, kết hợp LLM dịch thuật/quy đổi từ khóa đối với câu phức tạp"""
        if not text:
            return []
        
        syndromes = []
        try:
            # Kiểm tra nếu câu chứa tiếng Anh (từ LLaVA face description)
            english_indicators = ["patient", "pale", "complexion", "expression", "spirit", "eyes", "swelling", "spots", "rash", "skin", "face", "fatigue", "spiritless", "puffiness", "dark circles", "mole"]
            has_english = any(w in text.lower() for w in english_indicators)

            # 1. Thử khớp triệu chứng chính xác từ database trước (Longest Match First)
            exact_symptoms = self.qa_pipeline._preprocess_question(text)
            if exact_symptoms:
                logger.info(f"Đã khớp triệu chứng chính xác từ database: {exact_symptoms}")
                symptoms_lower = [s.lower() for s in exact_symptoms]
                cypher = f"""
                MATCH (h:HoiChung)-[:CÓ_BIỂU_HIỆN]->(t:TrieuChung)
                WHERE toLower(t.name) IN {symptoms_lower}
                RETURN DISTINCT h.name
                """
                records = self.qa_pipeline.run_cypher(cypher)
                for rec in records:
                    if rec.get('h.name'):
                        syndromes.append(rec['h.name'])

            # Thử khớp tên bệnh lý (BenhLy) từ database
            matched_diseases = []
            try:
                with self.qa_pipeline.driver.session() as session:
                    result = session.run("MATCH (b:BenhLy) RETURN b.name AS name")
                    db_diseases = [record["name"] for record in result]
                
                db_diseases_sorted = sorted(db_diseases, key=len, reverse=True)
                text_lower = text.lower()
                for disease in db_diseases_sorted:
                    if len(disease.split()) < 2 and disease.lower() not in ["ho", "lao"]:
                        continue
                    import re
                    pattern = rf'\b{re.escape(disease.lower())}\b'
                    if re.search(pattern, text_lower):
                        matched_diseases.append(disease)
            except Exception as ex:
                logger.error(f"Lỗi khớp tên bệnh lý: {ex}")

            # Lọc bỏ các bệnh lý trùng lặp hoặc là con (sub-phrase) của các triệu chứng đã khớp để ưu tiên triệu chứng đặc hiệu
            if exact_symptoms and matched_diseases:
                filtered_diseases = []
                symptoms_lower = [s.lower() for s in exact_symptoms]
                for disease in matched_diseases:
                    disease_lower = disease.lower()
                    if any(disease_lower in sym for sym in symptoms_lower):
                        continue
                    filtered_diseases.append(disease)
                matched_diseases = filtered_diseases

            if matched_diseases:
                logger.info(f"Đã khớp bệnh lý chính xác từ database: {matched_diseases}")
                diseases_lower = [d.lower() for d in matched_diseases]
                cypher = f"""
                MATCH (b:BenhLy)-[:CHIA_THÀNH]->(h:HoiChung)
                WHERE toLower(b.name) IN {diseases_lower}
                RETURN DISTINCT h.name
                """
                records = self.qa_pipeline.run_cypher(cypher)
                for rec in records:
                    if rec.get('h.name'):
                        syndromes.append(rec['h.name'])
                        
            # Nếu đã khớp được hội chứng từ database và KHÔNG có tiếng Anh, trả về ngay để tiết kiệm tài nguyên
            if syndromes and not has_english:
                return list(set(syndromes))

            # 2. Fallback hoặc có chứa tiếng Anh: dùng LLM dịch thuật và đóng vai trò màng lọc nhiễu
            word_count = len(text.split())
            should_run_llm = not syndromes or word_count > 10 or has_english
            
            if should_run_llm:
                if has_english:
                    logger.info("Phát hiện mô tả tiếng Anh hoặc cần kích hoạt LLM dịch thuật & trích xuất...")
                else:
                    logger.info("Kích hoạt LLM trích xuất và quy đổi triệu chứng từ câu dài/phức tạp...")
                    
                text_norm = normalize_symptoms_text(text)
                
                if has_english:
                    prompt = f"""
                    Câu của bệnh nhân (chứa triệu chứng tiếng Việt và mô tả sắc mặt bằng tiếng Anh): "{text_norm}"
                    Nhiệm vụ: Chỉ trích xuất các danh từ/động từ chỉ TRIỆU CHỨNG Y KHOA bằng TIẾNG VIỆT thực sự.
                    LƯU Ý QUAN TRỌNG: Nếu trong câu có mô tả bằng tiếng Anh (ví dụ: 'pale complexion', 'dark circles under eyes', 'spots on face', 'puffiness'), hãy DỊCH và quy đổi chúng sang triệu chứng Đông y tiếng Việt tương ứng (ví dụ: 'sắc mặt nhợt nhạt', 'quầng thâm mắt', 'mặt có ban', 'mặt phù').
                    MỖI TRIỆU CHỨNG TIẾNG VIỆT PHẢI CÓ TỪ 2 TỪ TRỞ LÊN (ví dụ: ho khan, ho đờm, đau đầu, sắc mặt nhợt).
                    TUYỆT ĐỐI KHÔNG trích xuất các triệu chứng chủ quan về tinh thần, cảm xúc, thần sắc hoặc biểu cảm khuôn mặt từ ảnh chụp (ví dụ: loại bỏ hoàn toàn các từ như 'mệt mỏi', 'thần sắc mệt mỏi', 'biểu cảm mệt mỏi', 'biểu cảm trung tính', 'dấu hiệu mệt mỏi', 'căng thẳng', 'làm việc quá sức'). Chỉ tập trung vào các đặc điểm thực thể vật lý thực sự (ví dụ: sắc mặt nhợt nhạt, quầng thâm mắt, mặt phù, lưỡi nhợt, rêu trắng...).
                    TUYỆT ĐỐI KHÔNG trích xuất các từ đơn lẻ chỉ có 1 từ (ví dụ: ho, sốt, đau, mỏi) và loại bỏ các trạng từ chỉ mức độ, thời gian hoặc từ xưng hô (ví dụ: tôi, bị, liên tục, nhiều, quá, rũ rượi, dồn dập).
                    Chỉ trả về danh sách các từ khóa tiếng Việt cốt lõi, cách nhau bằng dấu phẩy. Không giải thích gì thêm.
                    """
                else:
                    prompt = f"""
                    Câu của bệnh nhân: "{text_norm}"
                    Nhiệm vụ: Chỉ trích xuất các danh từ/động từ chỉ TRIỆU CHỨNG Y KHOA Đông y thực sự.
                    MỖI TRIỆU CHỨNG PHẢI CÓ TỪ 2 TỪ TRỞ LÊN (ví dụ: ho khan, ho đờm, đau đầu, sốt cao, nôn mửa, chóng mặt).
                    TUYỆT ĐỐI KHÔNG trích xuất các từ đơn lẻ chỉ có 1 từ (ví dụ: ho, sốt, đau, mỏi) và loại bỏ các trạng từ chỉ mức độ, thời gian hoặc từ xưng hô (ví dụ: tôi, bị, liên tục, nhiều, quá, rũ rượi, dồn dập).
                    
                    LƯU Ý QUAN TRỌNG: Nếu trong câu có mô tả cảm quan hoặc hình ảnh lâm sàng (ví dụ: 'tông màu da nhợt nhạt', 'mặt trông nhợt nhạt', 'da nhợt', 'vàng xanh quanh mắt'), hãy quy đổi/dịch chúng sang thuật ngữ triệu chứng y khoa chuẩn xác tương ứng (ví dụ: 'sắc mặt nhợt nhạt', 'mặt nhợt', 'sắc mặt vàng').
                    TUYỆT ĐỐI KHÔNG trích xuất các triệu chứng chủ quan về tinh thần, cảm xúc, thần sắc hoặc biểu cảm khuôn mặt thu được từ ảnh chụp (ví dụ: loại bỏ hoàn toàn các từ như 'mệt mỏi', 'thần sắc mệt mỏi', 'biểu cảm mệt mỏi', 'biểu cảm trung tính', 'dấu hiệu mệt mỏi', 'căng thẳng', 'làm việc quá sức'). Chỉ tập trung vào các đặc điểm thực thể vật lý thực sự (ví dụ: sắc mặt nhợt nhạt, sắc mặt vàng, quầng thâm mắt, lưỡi nhợt, rêu trắng...).
                    
                    Chỉ trả về danh sách các từ khóa triệu chứng chuẩn xác, cách nhau bằng dấu phẩy. Không giải thích gì thêm.
                    """
                    
                res = self.qa_pipeline.client.chat(
                    model=self.qa_pipeline.llm_model,
                    messages=[
                        {"role": "system", "content": "Bạn là trợ lý y khoa chỉ trích xuất từ khóa bằng tiếng Việt từ câu hỏi của bệnh nhân."},
                        {"role": "user", "content": prompt}
                    ],
                    options={"temperature": 0.0, "seed": 42}
                )
                keywords_str = res['message']['content'].strip()
                
                # Xóa bỏ các ký tự thừa nếu LLM lỡ sinh ra
                keywords_str = keywords_str.replace('"', '').replace("'", "").replace(".", "")
                
                # Tách thành mảng các từ khóa sạch (bắt buộc triệu chứng phải từ 2 từ trở lên, ví dụ: ho khan, ho đờm, đau đầu)
                keywords = [k.strip().lower() for k in keywords_str.split(',') if len(k.strip().split()) >= 2]
                
                # Loại bỏ các từ khóa từ LLM nếu chúng trùng hoặc là tập con/tập cha của triệu chứng khớp chính xác từ database
                if exact_symptoms:
                    symptoms_lower = [sym.lower() for sym in exact_symptoms]
                    filtered_keywords = []
                    for kw in keywords:
                        overlap = False
                        for sym_l in symptoms_lower:
                            if kw in sym_l or sym_l in kw:
                                overlap = True
                                break
                        if not overlap:
                            filtered_keywords.append(kw)
                    keywords = filtered_keywords
                    
                    # Giữ nguyên các triệu chứng khớp chính xác từ database làm từ khóa chính
                    for sym_l in symptoms_lower:
                        if sym_l not in keywords:
                            keywords.append(sym_l)
                            
                logger.info(f"Từ khóa y khoa đã lọc sạch nhiễu và quy đổi: {keywords}")
                
                # Quét trực tiếp Database với các từ khóa
                for kw in keywords:
                    cypher = f"""
                    MATCH (h:HoiChung)-[:CÓ_BIỂU_HIỆN]->(t:TrieuChung) 
                    WHERE toLower(t.name) CONTAINS '{kw}' 
                    RETURN DISTINCT h.name
                    """
                    records = self.qa_pipeline.run_cypher(cypher)
                    for rec in records:
                        if rec.get('h.name'):
                            syndromes.append(rec['h.name'])
                            
            return list(set(syndromes))
        except Exception as e:
            logger.error(f"Lỗi khi trích xuất hội chứng: {e}")
            return []

    def _get_face_tongue_symptoms_for_syndrome(self, syndrome: str) -> dict:
        tongue_symptoms = []
        face_symptoms = []
        try:
            import json
            import os
            
            tongue_file = "data/mapping/symptom_to_syndrome.json"
            if os.path.exists(tongue_file):
                with open(tongue_file, "r", encoding="utf-8") as f:
                    tongue_map = json.load(f)
                for symptom, syndromes in tongue_map.items():
                    if syndrome in syndromes or any(syndrome.lower() == s.lower() for s in syndromes):
                        if symptom not in tongue_symptoms:
                            tongue_symptoms.append(symptom)
                            
            face_file = "data/mapping/face_to_syndrome.json"
            if os.path.exists(face_file):
                with open(face_file, "r", encoding="utf-8") as f:
                    face_map = json.load(f)
                for symptom, syndromes in face_map.items():
                    if syndrome in syndromes or any(syndrome.lower() == s.lower() for s in syndromes):
                        if symptom not in face_symptoms:
                            face_symptoms.append(symptom)
        except Exception as e:
            logger.error(f"Lỗi đọc file mapping: {e}")
            
        if not tongue_symptoms and not face_symptoms:
            try:
                symptoms = self.vision_pipeline.neo4j_client.get_symptoms_by_syndrome(syndrome)
                for s in symptoms:
                    if "lưỡi" in s.lower() or "rêu" in s.lower():
                        tongue_symptoms.append(s)
                    elif "mặt" in s.lower() or "sắc" in s.lower():
                        face_symptoms.append(s)
                    else:
                        tongue_symptoms.append(s)
            except Exception as e:
                logger.error(f"Lỗi lấy triệu chứng Neo4j: {e}")
                
        if not tongue_symptoms: tongue_symptoms = ["Lưỡi nhợt", "Lưỡi bệu có dấu răng"]
        if not face_symptoms: face_symptoms = ["Mặt nhợt nhạt"]
            
        return {"tongue": tongue_symptoms, "face": face_symptoms}

    def _filter_syndromes_with_llm(self, symptoms: list, candidate_syndromes: list) -> list:
        if not candidate_syndromes: return []
        if len(candidate_syndromes) <= 1: return candidate_syndromes
            
        symptoms_str = ", ".join(symptoms)
        candidates_str = ", ".join(candidate_syndromes)
        
        prompt = f"""
        Bệnh nhân có triệu chứng: {symptoms_str}. Các hội chứng dự kiến: {candidates_str}.
        Hãy chọn ra từ 1 đến 3 hội chứng chính xác nhất và trả về dưới dạng danh sách ngăn cách bởi dấu phẩy. Không giải thích gì thêm.
        """
        try:
            response = self.qa_pipeline.client.chat(
                model=self.qa_pipeline.llm_model,
                messages=[
                    {"role": "system", "content": "Bạn là bác sĩ Đông y chỉ phản hồi bằng tiếng Việt."},
                    {"role": "user", "content": prompt}
                ],
                options={"temperature": 0.0, "seed": 42}
            )
            ans = response['message']['content'].strip()
            selected = [s.strip() for s in ans.replace("\n", ",").split(",") if s.strip()]
            valid_selected = [s for s in selected if any(s.lower() == cand.lower() for cand in candidate_syndromes)]
            
            if valid_selected:
                final = []
                for v in valid_selected:
                    for cand in candidate_syndromes:
                        if v.lower() == cand.lower() and cand not in final:
                            final.append(cand)
                return final
            return candidate_syndromes
        except Exception as e:
            logger.error(f"Lỗi LLM filter: {e}")
            return candidate_syndromes

    def _generate_explainable_answer(self, user_symptoms: str, detected_symptoms: list, detailed_kg_data: list) -> str:
        """
        [Giải pháp Chống Ảo Giác Tuyệt Đối]
        Sử dụng Python để render cấu trúc thực thể (Tên bệnh, Bài thuốc) 100% từ Neo4j.
        Chỉ gọi LLM Qwen để sinh đoạn văn giải thích cơ chế bệnh sinh.
        """
        final_markdown = "### 1. Kết luận chẩn đoán\n"
        all_syndromes = [item["syndrome"] for item in detailed_kg_data]
        final_markdown += f"- **Các hội chứng liên quan:** {', '.join(all_syndromes)}\n"
        
        # Xác định hội chứng khả quan nhất và liên quan nhất
        if len(all_syndromes) > 1:
            symptoms_str = user_symptoms
            if detected_symptoms:
                symptoms_str = f"{symptoms_str}, {', '.join(detected_symptoms)}"
            
            prompt_primary = f"""
            Vai trò: Bác sĩ Đông y giàu kinh nghiệm.
            Bệnh nhân có các triệu chứng: {symptoms_str}.
            Danh sách các hội chứng được tìm thấy: {', '.join(all_syndromes)}.
            Nhiệm vụ: Hãy chọn ra đúng 1 HỘI CHỨNG KHẢ QUAN VÀ LIÊN QUAN NHẤT (hội chứng cốt lõi/phù hợp nhất với các triệu chứng của bệnh nhân) từ danh sách trên, và viết đúng 1-2 câu giải thích ngắn gọn lý do tại sao hội chứng đó là phù hợp nhất.
            Yêu cầu: Viết hoàn toàn bằng tiếng Việt tự nhiên sạch sẽ. Không dùng chữ Hán hay tiếng Trung. Chỉ trả về kết quả định dạng:
            - **Hội chứng khả quan nhất:** [Tên hội chứng]
            - **Lý do phù hợp:** [Giải thích ngắn gọn 1-2 câu]
            Tuyệt đối không thêm bất kỳ lời chào, markdown phụ hay giải thích nào khác ngoài định dạng trên.
            """
            try:
                response = self.qa_pipeline.client.chat(
                    model=self.qa_pipeline.llm_model,
                    messages=[
                        {"role": "system", "content": "Bạn là bác sĩ Đông y Việt Nam. Bạn chỉ phản hồi kết luận hội chứng phù hợp nhất theo đúng định dạng được yêu cầu."},
                        {"role": "user", "content": prompt_primary}
                    ],
                    options={"temperature": 0.1, "seed": 42}
                )
                primary_text = response['message']['content'].strip()
                primary_text = self._clean_foreign_characters(primary_text)
                final_markdown += f"{primary_text}\n\n"
            except Exception as e:
                logger.error(f"Lỗi chọn hội chứng chính: {e}")
                final_markdown += f"- **Hội chứng khả quan nhất:** {all_syndromes[0]}\n- **Lý do phù hợp:** Đây là hội chứng có độ tương thích cao với tổ hợp triệu chứng của bệnh nhân.\n\n"
        else:
            final_markdown += f"- **Hội chứng khả quan nhất:** {all_syndromes[0]}\n- **Lý do phù hợp:** Đây là hội chứng duy nhất phù hợp hoàn toàn với các biểu hiện lâm sàng của bệnh nhân.\n\n"
        
        final_markdown += "### 2. Biện chứng luận trị và Đồ thị tri thức\n"
        
        for data in detailed_kg_data:
            syndrome = data["syndrome"]
            matching = data["matching_symptoms"]
            diseases = data["diseases"]
            bai_thuoc = data["bai_thuoc"]
            vi_thuoc = data["vi_thuoc"]
            
            final_markdown += f"#### Hội chứng: {syndrome}\n"
            
            # Chỉ dùng LLM để giải thích, CẤM kê đơn
            prompt = f"""
            Vai trò: Bác sĩ Đông y.
            Bệnh nhân có hội chứng: '{syndrome}'. 
            Các triệu chứng: {', '.join(matching) if matching else user_symptoms}.
            Nhiệm vụ: Viết ĐÚNG 1 đoạn văn (3-4 câu) giải thích cơ chế y lý theo Âm Dương Ngũ Hành tại sao các triệu chứng trên lại gây ra hội chứng '{syndrome}'.
            YÊU CẦU NGÔN NGỮ: BẮT BUỘC viết hoàn toàn bằng tiếng Việt tự nhiên. TUYỆT ĐỐI KHÔNG sử dụng bất kỳ chữ Hán hay tiếng Trung Quốc nào (ví dụ: không được viết 阳, 阴, 清淡饮食... mà phải dịch hoặc giải thích rõ bằng tiếng Việt).
            LỆNH CẤM THÉP: TUYỆT ĐỐI KHÔNG kê đơn thuốc, KHÔNG nhắc đến tên bài thuốc hay vị thuốc nào, KHÔNG đưa ra lời khuyên. CHỈ giải thích cơ chế.
            """
            try:
                response = self.qa_pipeline.client.chat(
                    model=self.qa_pipeline.llm_model,
                    messages=[
                        {"role": "system", "content": "Bạn là bác sĩ Đông y Việt Nam uyên bác. Bạn chỉ viết hoàn toàn bằng chữ cái tiếng Việt. Bạn tuyệt đối không được viết bất kỳ chữ Hán hay chữ Trung Quốc nào, kể cả các thuật ngữ y khoa (ví dụ: không viết 阴, 阳, 稀释, 保持...). Mọi thuật ngữ phải được viết bằng tiếng Việt thuần túy."},
                        {"role": "user", "content": prompt}
                    ],
                    options={"temperature": 0.1, "seed": 42}
                )
                explanation = response['message']['content'].strip()
            except Exception as e:
                logger.error(f"Lỗi LLM giải thích: {e}")
                explanation = "Không thể sinh lời giải thích y lý."
                
            final_markdown += f"- **Lý giải Y lý:** {explanation}\n"
            
            # Trích dẫn thực thể thẳng từ Neo4j bằng Python
            if diseases:
                final_markdown += f"- **Bệnh lý liên quan:** Nhóm bệnh **{', '.join(diseases)}** `(BenhLy)-[:CHIA_THÀNH]->(HoiChung)`\n"
            
            if bai_thuoc != "Chưa có dữ liệu":
                final_markdown += f"- **Phương dược điều trị:** Bài thuốc **{bai_thuoc}** `(HoiChung)-[:ĐƯỢC_ĐIỀU_TRỊ_BẰNG]->(BaiThuoc)`\n"
                final_markdown += f"- **Thành phần vị thuốc:** Bao gồm **{vi_thuoc}** `(BaiThuoc)-[:BAO_GỒM]->(ViThuoc)`\n\n"
            else:
                final_markdown += f"- **Phương dược điều trị:** Hiện chưa có bài thuốc cập nhật cho hội chứng này trong hệ thống.\n\n"
        # Thu thập danh sách tất cả các bệnh để sinh lời khuyên phù hợp
        all_diseases = []
        for data in detailed_kg_data:
            if data["diseases"]:
                all_diseases.extend(data["diseases"])
        all_diseases = list(set(all_diseases))

        explanation_advice = ""
        if all_diseases:
            prompt_advice = f"""
            Vai trò: Bác sĩ Đông y.
            Bệnh nhân có các bệnh lý liên quan: {', '.join(all_diseases)}.
            Nhiệm vụ: Viết 2-3 lời khuyên ngắn gọn (gạch đầu dòng), thiết thực về chế độ ăn uống, sinh hoạt, nghỉ ngơi cụ thể cho nhóm bệnh trên.
            YÊU CẦU NGÔN NGỮ: BẮT BUỘC viết hoàn toàn bằng tiếng Việt. TUYỆT ĐỐI KHÔNG sử dụng tiếng Trung Quốc hoặc chữ Hán trong các lời khuyên.
            Lưu ý: Viết súc tích, dễ hiểu, chuyên nghiệp. Không ghi lời mở đầu hay kết bài, chỉ trả về các dòng gạch đầu dòng.
            """
            try:
                res_adv = self.qa_pipeline.client.chat(
                    model=self.qa_pipeline.llm_model,
                    messages=[
                        {"role": "system", "content": "Bạn là bác sĩ Đông y Việt Nam uyên bác. Bạn chỉ viết lời khuyên bằng chữ cái tiếng Việt. Bạn tuyệt đối không sử dụng chữ Hán hay chữ Trung Quốc (như 稀释, 保持...), tất cả lời khuyên phải dùng tiếng Việt thuần túy."},
                        {"role": "user", "content": prompt_advice}
                    ],
                    options={"temperature": 0.2, "seed": 42}
                )
                explanation_advice = res_adv['message']['content'].strip()
            except Exception as e:
                logger.error(f"Lỗi LLM sinh lời khuyên: {e}")

        if not explanation_advice:
            explanation_advice = (
                "- Vui lòng nghỉ ngơi điều độ, tránh căng thẳng và giữ tinh thần thoải mái.\n"
                "- Ăn uống thanh đạm, hạn chế thực phẩm nhiều dầu mỡ, cay nóng hoặc khó tiêu.\n"
                "- Theo dõi sát sao các triệu chứng và đến cơ sở y tế gần nhất nếu có dấu hiệu bất thường."
            )

        final_markdown += "### 3. Lời khuyên tổng quát\n"
        final_markdown += explanation_advice

        return self._clean_foreign_characters(final_markdown)

    def run_diagnosis(self, user_symptoms: str = "", face_img_path: str = None, tongue_img_path: str = None) -> dict:
        vision_analysis_text = ""
        raw_vision_data = None

        if face_img_path or tongue_img_path:
            logger.info("Bắt đầu phân tích hình ảnh qua LLaVA...")
            try:
                raw_vision_data = self.vision_pipeline.run(tongue_image_path=tongue_img_path, face_image_path=face_img_path)
                if raw_vision_data and isinstance(raw_vision_data, dict):
                    tongue_desc = raw_vision_data.get("tongue_description", "")
                    face_desc = raw_vision_data.get("face_description", "")
                    
                    detected_symptoms = []
                    
                    if tongue_desc:
                        logger.info(f"Khớp mô tả lưỡi với danh sách triệu chứng chuẩn...")
                        mapped_tongue = self._map_desc_to_symptoms(tongue_desc, self.tongue_symptoms_list)
                        logger.info(f"Triệu chứng lưỡi đã khớp: {mapped_tongue}")
                        detected_symptoms.extend(mapped_tongue)
                        
                    if face_desc:
                        logger.info(f"Khớp mô tả sắc mặt với danh sách triệu chứng chuẩn...")
                        mapped_face = self._map_desc_to_symptoms(face_desc, self.face_symptoms_list)
                        logger.info(f"Triệu chứng mặt đã khớp: {mapped_face}")
                        detected_symptoms.extend(mapped_face)
                        
                    # Nếu không khớp được triệu chứng nào qua LLM mapping, tiến hành dịch và thử trích xuất bằng preprocess cũ
                    if not detected_symptoms:
                        logger.info("Không khớp được triệu chứng chuẩn nào. Fallback sang dịch mô tả y khoa...")
                        translated_symptoms = []
                        for s in raw_vision_data.get("detected_symptoms", []):
                            if any(w in s.lower() for w in ["patient", "pale", "complexion", "expression", "spirit", "eyes", "swelling", "spots", "rose", "skin", "face", "fatigue", "spiritless", "puffiness", "dark circles", "mole", "tongue", "coating", "fur", "body", "tooth marks", "scalloped", "swollen", "cracked", "red", "purple", "slipperiness", "greasy", "peeled"]):
                                logger.info(f"Dịch mô tả tiếng Anh sang tiếng Việt: {s[:50]}...")
                                translated_s = self._translate_english_description(s)
                                translated_symptoms.append(translated_s)
                            else:
                                translated_symptoms.append(s)
                        
                        fallback_extracted = []
                        for ts in translated_symptoms:
                            extracted = self.qa_pipeline._preprocess_question(ts)
                            fallback_extracted.extend(extracted)
                            
                        if fallback_extracted:
                            detected_symptoms = list(set(fallback_extracted))
                            logger.info(f"Fallback trích xuất triệu chứng từ bản dịch: {detected_symptoms}")
                        else:
                            detected_symptoms = translated_symptoms
                    else:
                        # Loại bỏ trùng lặp
                        detected_symptoms = list(set(detected_symptoms))
                            
                    raw_vision_data["detected_symptoms"] = detected_symptoms
                    vision_analysis_text = ", ".join(detected_symptoms)
                    raw_vision_data["analysis"] = vision_analysis_text
            except Exception as e:
                logger.error(f"Lỗi module Vision: {e}")

        combined_query = user_symptoms.strip()
        if vision_analysis_text:
            combined_query = f"{combined_query}, {vision_analysis_text}" if combined_query else vision_analysis_text

        final_syndromes = self._extract_syndromes_from_text(combined_query) if combined_query else []

        if len(final_syndromes) > 1:
            symptoms_list = []
            if user_symptoms: symptoms_list.append(user_symptoms)
            if raw_vision_data and isinstance(raw_vision_data, dict):
                symptoms_list.extend(raw_vision_data.get("detected_symptoms", []))
            if symptoms_list:
                final_syndromes = self._filter_syndromes_with_llm(symptoms_list, final_syndromes)

        # === BẮT ĐẦU CHẶN KẾT QUẢ KHÔNG HỢP LỆ ===
        # Danh sách các hội chứng hợp lệ (bạn đã có trong prompts.py, copy vào đây)
        VALID_SYNDROMES = [
            "Can Thận hư khuy", "Can dương huyết ứ", "Can dương hóa phong", "Can dương thượng kháng", 
            "Can huyết bất túc", "Can hỏa", "Can hỏa cuồng động", "Can hỏa nhiễu tâm", "Can hỏa phạm phế", 
            "Can hỏa thượng viêm", "Can khí phạm vị", "Can khí uất kết", "Can khí uất trệ", "Can thận hư", 
            "Can thận khuy hư", "Can thận âm hư", "Can uất", "Can uất huyết ứ", "Can uất hóa hỏa", 
            "Can uất hỏa", "Can uất hỏa uất", "Can uất hỏa vượng", "Can uất khí trệ", "Can đảm hỏa thịnh", 
            "Can đảm thấp nhiệt", "Cơ ho dữ dội", "Dinh phận chứng", "Dương hoàng - Nhiệt trọng", 
            "Dương hoàng - Thấp trọng", "Dương hư bí", "Dịch độc lị", "Gan thận âm hư", "Hoàng đản", 
            "Huyết hàn", "Huyết hư", "Huyết hư bí", "Huyết hư phong táo", "Huyết khô", "Huyết lâm", 
            "Huyết nhiệt", "Huyết nhiệt vọng hành", "Huyết nhiệt ứ", "Huyết phận chứng", "Huyết ứ", 
            "Huyết ứ mạch lạc", "Huyết ứ nội kết", "Huyết ứ trở lạc", "Hàn hoắc loạn", "Hàn ngưng", 
            "Hàn ngưng mạch lạc", "Hàn ngưng tâm mạch", "Hàn ngưng uất kết", "Hàn sán", "Hàn thấp", 
            "Hàn thấp lị", "Hàn thấp nội thịnh", "Hàn thấp trở lạc", "Hàn thấp uất kết", "Hàn thấp úc kết", 
            "Hàn tà khách vị", "Hành tý (Phong tý)", "Hư bổn lị", "Hư hỏa", "Khí bí", "Khí bất nhiếp huyết", 
            "Khí dương bạo thoát", "Khí huyết câu hư", "Khí huyết hư", "Khí huyết khuy hư", "Khí huyết lưỡng hư", 
            "Khí hư", "Khí hư bí", "Khí hư huyết ứ", "Khí lâm", "Khí phận chứng", "Khí trệ", "Khí trệ hung", 
            "Khí trệ huyết ứ", "Khí trệ thấp trở", "Khí trệ tâm hung", "Khí trệ uất kết", "Khí uất hóa hỏa", 
            "Khí âm lưỡng hư", "Khởi phát", "Kinh lạc - Khí hư huyết ứ", "Kinh lạc - Phong đàm", "Lao lâm", 
            "Loét mủ", "Mệnh môn hỏa suy", "Ngoại tà phạm vị", "Nhiệt bí", "Nhiệt hoắc loạn", "Nhiệt lâm", 
            "Nhiệt thấp", "Nhiệt thịnh", "Nhiệt tý", "Nhiệt độc thịnh", "Nhiệt độc uất kết", "Nhiệt độc xí thịnh", 
            "Phong dương nội động", "Phong hàn", "Phong hàn phạm phế", "Phong hàn thấp", "Phong hàn trở lạc", 
            "Phong hỏa thịnh", "Phong nhiệt", "Phong nhiệt huyết táo", "Phong nhiệt phạm phế", "Phong nhiệt thấp", 
            "Phong nhiệt thịnh", "Phong nhiệt trở lạc", "Phong thấp", "Phong thấp nhiệt", "Phong thấp trở lạc", 
            "Phong thủy tướng bác", "Phong tà trúng lạc", "Phong táo thương phế", "Phong đàm bế khiếu", 
            "Phong đàm nội nhiễu", "Phong đàm trở lạc", "Phế khí hư", "Phế nhiệt", "Phế nhiệt di tân", 
            "Phế nhiệt thịnh", "Phế nhiệt tân thương", "Phế nhiệt uất kết", "Phế phế khí hư", "Phế vệ bất cố", 
            "Phế vị nhiệt thịnh", "Phế âm hư", "Thành mủ", "Thạch lâm", "Thấp hàn", "Thấp nhiệt", 
            "Thấp nhiệt bọc mủ", "Thấp nhiệt hạ chú", "Thấp nhiệt lị", "Thấp nhiệt sán", "Thấp nhiệt trở lạc", 
            "Thấp nhiệt tẩm dâm", "Thấp nhiệt uẩn kết", "Thấp nhiệt uẩn phế", "Thấp trệ", "Thấp trọng", 
            "Thấp trở", "Thấp độc tẩm dâm", "Thận bất nạp khí", "Thận dương hư", "Thận hư", "Thận hư bất cố", 
            "Thận hư tinh khuy", "Thận hư đàm thấp", "Thận khí hư", "Thận tinh bất túc", "Thận tinh hư", 
            "Thận âm bất túc", "Thận âm hư", "Thận âm hư hoả vượng", "Thận âm khuy hư", "Thống tý (Hàn tý)", 
            "Thời kỳ sởi bay", "Thời kỳ sởi mọc", "Thủy thấp nội đình", "Thủy thấp tẩm trướng", "Thử thấp", 
            "Thực trệ", "Thực trệ tràng vị", "Thực trệ uất kết", "Thực trệ uẩn kết", "Thực trệ ứ", 
            "Thực độc trúng", "Trùng tích", "Tr착 tý (Thấp tý)", "Táo nhiệt thương phế", "Tâm dương bạo thoát", 
            "Tâm dương bất chấn", "Tâm dương hư", "Tâm huyết bất túc", "Tâm huyết ứ trở", "Tâm hỏa", 
            "Tâm hỏa thượng viêm", "Tâm hỏa thịnh", "Tâm khí bất túc", "Tâm khí dương suy", "Tâm khí hư", 
            "Tâm thần thất dưỡng", "Tâm thận bất giao", "Tâm thận khuy hư", "Tâm tỳ hư", "Tâm tỳ khuy hư", 
            "Tâm tỳ lưỡng hư", "Tâm đởm khí hư", "Tân dịch thương", "Tạng phủ - Bế chứng", "Tạng phủ - Thoát chứng", 
            "Tỳ dương hư", "Tỳ hư thấp trọng", "Tỳ khí hư", "Tỳ thận dương hư", "Tỳ thận hư", "Tỳ thận khí hư", 
            "Tỳ vị hư", "Tỳ vị hư hàn", "Tỳ vị hư nhược", "Tỳ vị hư đàm thấp", "Tỳ vị khí hư", "Tỳ vị thấp nhiệt", 
            "Vệ phận chứng", "Vị hàn", "Vị hỏa thượng viêm", "Vị nhiệt", "Vị nhiệt xí thịnh", "Vị âm bất túc", 
            "Vị âm hư", "Âm dương lưỡng hư", "Âm hoàng", "Âm hư", "Âm hư dương kháng", "Âm hư hỏa uất", 
            "Âm hư hỏa vượng", "Âm hư nội nhiệt", "Đàm hỏa bế khiếu", "Đàm hỏa nhiễu tâm", "Đàm hỏa nội nhiễu", 
            "Đàm hỏa thượng nhiễu", "Đàm khí giao trở", "Đàm khí uất kết", "Đàm khí uất kết (Mai hạch khí)", 
            "Đàm nhiệt", "Đàm nhiệt nội nhiễu", "Đàm nhiệt thịnh", "Đàm thấp", "Đàm thấp bế khiếu", 
            "Đàm thấp nội nhiễu", "Đàm thấp thủy dâm", "Đàm thấp trở lạc", "Đàm thấp uất kết", "Đàm thấp uẩn phế", 
            "Đàm trọc bế trở", "Đàm trọc bế tắc", "Đàm trọc trung trở", "Đàm ứ trở lạc", "Đàm ứ tý (Lâu ngày)", 
            "Ẩm thực đình trệ", "Ứ huyết", "Ứ huyết nội kết", "Ứ huyết trở lạc", "Ứ trệ kỳ đầu"
        ]

        # Lọc lại final_syndromes, chỉ giữ các hội chứng có trong danh sách hợp lệ
        filtered_final_syndromes = []
        # Tạo mapping case-insensitive từ self.valid_syndromes và VALID_SYNDROMES
        valid_syndromes_map = {vs.lower(): vs for vs in self.valid_syndromes} if hasattr(self, 'valid_syndromes') and self.valid_syndromes else {}
        fallback_syndromes_map = {vs.lower(): vs for vs in VALID_SYNDROMES}

        for s in final_syndromes:
            s_lower = s.lower()
            if valid_syndromes_map:
                if s_lower in valid_syndromes_map:
                    filtered_final_syndromes.append(valid_syndromes_map[s_lower])
                else:
                    logger.warning(f"Phát hiện hội chứng không hợp lệ (ảo giác): '{s}', đã loại bỏ")
            else:
                # Fallback dùng danh sách hardcoded
                if s_lower in fallback_syndromes_map:
                    filtered_final_syndromes.append(fallback_syndromes_map[s_lower])
                else:
                    logger.warning(f"Phát hiện hội chứng không hợp lệ (ảo giác): '{s}', đã loại bỏ")

        # Nếu sau khi lọc không còn hội chứng nào, đặt lại final_syndromes thành mảng rỗng
        if not filtered_final_syndromes and final_syndromes:
            logger.error("Tất cả các hội chứng được tìm thấy đều là ảo giác. Hủy kết quả.")
            final_syndromes = []
        else:
            final_syndromes = filtered_final_syndromes
            
        # Loại bỏ các hội chứng trùng lặp giữ nguyên thứ tự
        seen = set()
        final_syndromes = [x for x in final_syndromes if not (x in seen or seen.add(x))]
        # === KẾT THÚC CHẶN KẾT QUẢ KHÔNG HỢP LỆ ===

        qa_result = None
        if final_syndromes:
            import json, os
            all_mappings = {}
            for filename in ["symptom_to_syndrome.json", "face_to_syndrome.json"]:
                path = os.path.join("data", "mapping", filename)
                if os.path.exists(path):
                    try:
                        with open(path, "r", encoding="utf-8") as f:
                            all_mappings.update(json.load(f))
                    except Exception as e: pass

            patient_symptoms = []
            if user_symptoms: patient_symptoms.extend([s.strip() for s in user_symptoms.split(",") if s.strip()])
            if raw_vision_data and isinstance(raw_vision_data, dict):
                patient_symptoms.extend(raw_vision_data.get("detected_symptoms", []))
            patient_symptoms = list(set([s for s in patient_symptoms if s]))

            treatments = []
            detailed_kg_data = [] 

            for syndrome in final_syndromes:
                matching_patient_symptoms = []
                for s in patient_symptoms:
                    mapped_syndromes = all_mappings.get(s, [])
                    if syndrome in mapped_syndromes or any(syndrome.lower() == ms.lower() for ms in mapped_syndromes):
                        matching_patient_symptoms.append(s)
                
                try:
                    diseases = self.qa_pipeline.neo4j_client.get_diseases_by_syndrome(syndrome)
                    
                    # Tìm bệnh lý phù hợp nhất trong số các bệnh lý của hội chứng này từ combined_query
                    target_disease = None
                    if diseases:
                        for d in diseases:
                            if d.lower() in combined_query.lower():
                                target_disease = d
                                break
                        if not target_disease:
                            for d in diseases:
                                d_clean = d.lower().replace("bệnh", "").strip()
                                if d_clean and d_clean in combined_query.lower():
                                    target_disease = d
                                    break
                    
                    treatment = self.qa_pipeline.neo4j_client.get_treatment_by_syndrome(syndrome, target_disease)
                    if not treatment and target_disease:
                        # Fallback nếu không có bài thuốc đặc hiệu cho bệnh lý đó
                        treatment = self.qa_pipeline.neo4j_client.get_treatment_by_syndrome(syndrome)
                    
                    bai_thuoc = "Chưa có dữ liệu"
                    vi_thuoc = "Chưa có dữ liệu"
                    
                    if treatment:
                        treatments.append(treatment)
                        bai_thuoc = treatment.get("bai_thuoc", "Chưa có dữ liệu")
                        vi_thuoc = ", ".join(treatment.get("vi_thuoc", [])) if treatment.get("vi_thuoc") else "Chưa có dữ liệu"

                    detailed_kg_data.append({
                        "syndrome": syndrome,
                        "matching_symptoms": matching_patient_symptoms,
                        "diseases": diseases,
                        "bai_thuoc": bai_thuoc,
                        "vi_thuoc": vi_thuoc
                    })
                except Exception as e:
                    logger.error(f"Lỗi truy xuất dữ liệu: {e}")
                    
            # CHỐT CHẶN ẢO GIÁC TUYỆT ĐỐI BẰNG PYTHON
            if not treatments:
                logger.warning(f"Neo4j không có bài thuốc cho {final_syndromes}")
                explainable_answer = (
                    f"### 1. Kết luận chẩn đoán sơ bộ\n"
                    f"Hệ thống nhận diện dấu hiệu liên quan đến hội chứng: **{', '.join(final_syndromes) if final_syndromes else 'Chưa rõ'}**.\n\n"
                    f"### 2. Phương dược điều trị\n"
                    f"Hiện tại, cơ sở dữ liệu Đồ thị Tri thức (Neo4j) **KHÔNG CÓ** bài thuốc tương ứng cho hội chứng này.\n\n"
                    f"> ⚠️ *Hệ thống đã tự động khóa mô hình AI sinh đơn thuốc để đảm bảo an toàn y tế, triệt tiêu hoàn toàn rủi ro AI tự bịa thuốc (Hallucination).* \n\n"
                    f"Vui lòng thử mô tả triệu chứng bằng các từ khóa khác."
                )
            else:
                detected_symptoms = raw_vision_data.get("detected_symptoms", []) if raw_vision_data and isinstance(raw_vision_data, dict) else []
                explainable_answer = self._generate_explainable_answer(
                    user_symptoms=user_symptoms,
                    detected_symptoms=detected_symptoms,
                    detailed_kg_data=detailed_kg_data
                )
                
            qa_result = {
                "answer": explainable_answer,
                "data": treatments,
                "syndromes": final_syndromes
            }
        else:
            qa_result = {
                "answer": "Hệ thống chưa tìm thấy bệnh lý hoặc hội chứng phù hợp với tổ hợp triệu chứng của bạn.",
                "data": []
            }

        return {
            "source": "Tứ chẩn hợp tham (Fusion)",
            "input_fusion": combined_query,
            "vision_details": raw_vision_data,
            "diagnosis_result": qa_result
        }

    def close(self):
        if hasattr(self, 'qa_pipeline'): self.qa_pipeline.close()
        if hasattr(self, 'vision_pipeline') and hasattr(self.vision_pipeline, 'close'): self.vision_pipeline.close()
        logger.info("Đã giải phóng tài nguyên.")
