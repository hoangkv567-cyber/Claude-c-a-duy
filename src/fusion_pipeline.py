# src/fusion_pipeline.py
import logging
from src.pipeline import TCMTonguePipeline
from src.qa_system import TCMQA

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
        logger.info("Khởi tạo hoàn tất!")

    def _extract_syndromes_from_text(self, text: str) -> list:
        """Trích xuất hội chứng bằng cách query trực tiếp Neo4j, đảm bảo không trượt dữ liệu"""
        if not text:
            return []
        syndromes = []
        try:
            # 1. Bóc tách từ khóa (VD: "tôi bị đau đầu" -> "đau đầu")
            terms = self.qa_pipeline._preprocess_question(text)
            if not terms:
                return []
            
            # 2. Quét trực tiếp Database bằng Cypher tĩnh (Không qua LLM)
            for term in terms:
                cypher = f"""
                MATCH (h:HoiChung)-[:CÓ_BIỂU_HIỆN]->(t:TrieuChung) 
                WHERE toLower(t.name) CONTAINS '{term.lower()}' 
                RETURN DISTINCT h.name
                """
                records = self.qa_pipeline.run_cypher(cypher)
                for rec in records:
                    if rec.get('h.name'):
                        syndromes.append(rec['h.name'])
                        
            return list(set(syndromes))
        except Exception as e:
            logger.error(f"Lỗi khi trích xuất hội chứng trực tiếp từ Neo4j: {e}")
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
            import ollama
            response = ollama.chat(
                model=self.qa_pipeline.llm_model,
                messages=[{"role": "user", "content": prompt}],
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
        final_markdown += f"Dựa trên phương pháp Tứ chẩn hợp tham, hệ thống chẩn đoán bệnh nhân có biểu hiện của hội chứng: **{', '.join(all_syndromes)}**.\n\n"
        
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
            LỆNH CẤM THÉP: TUYỆT ĐỐI KHÔNG kê đơn thuốc, KHÔNG nhắc đến tên bài thuốc hay vị thuốc nào, KHÔNG đưa ra lời khuyên. CHỈ giải thích cơ chế.
            """
            try:
                import ollama
                response = ollama.chat(
                    model=self.qa_pipeline.llm_model,
                    messages=[{"role": "user", "content": prompt}],
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
                
        final_markdown += "### 3. Lời khuyên tổng quát\n"
        final_markdown += "- Vui lòng nghỉ ngơi điều độ, ăn uống các thực phẩm dễ tiêu hóa và giữ tinh thần thoải mái. Theo dõi thêm triệu chứng để có hướng điều trị kịp thời."

        return final_markdown

    def run_diagnosis(self, user_symptoms: str = "", face_img_path: str = None, tongue_img_path: str = None) -> dict:
        vision_analysis_text = ""
        raw_vision_data = None
        vision_syndromes = []

        if face_img_path or tongue_img_path:
            logger.info("Bắt đầu phân tích hình ảnh qua LLaVA...")
            try:
                raw_vision_data = self.vision_pipeline.run(tongue_image_path=tongue_img_path, face_image_path=face_img_path)
                if raw_vision_data and isinstance(raw_vision_data, dict):
                    vision_analysis_text = raw_vision_data.get("analysis", "")
            except Exception as e:
                logger.error(f"Lỗi module Vision: {e}")

        if raw_vision_data and isinstance(raw_vision_data, dict):
            vision_syndromes = raw_vision_data.get("syndromes", raw_vision_data.get("possible_syndromes", []))

        text_syndromes = self._extract_syndromes_from_text(user_symptoms) if user_symptoms else []

        final_syndromes = []
        if text_syndromes and vision_syndromes:
            final_syndromes = list(set(text_syndromes) & set(vision_syndromes))
        elif text_syndromes:
            final_syndromes = text_syndromes
        elif vision_syndromes:
            final_syndromes = vision_syndromes

        combined_query = user_symptoms.strip()
        if vision_analysis_text:
            combined_query = f"{combined_query}, {vision_analysis_text}" if combined_query else vision_analysis_text

        if not final_syndromes and combined_query:
            try:
                qa_res = self.qa_pipeline.execute_and_answer(combined_query)
                extracted_syndromes = []
                if qa_res and "data" in qa_res:
                    for record in qa_res["data"]:
                        for k, v in record.items():
                            if ("h.name" in k or "hoichung" in k.lower() or "syndrome" in k.lower()) and v:
                                extracted_syndromes.append(v)
                if extracted_syndromes:
                    final_syndromes = list(set(extracted_syndromes))
            except Exception as e:
                logger.error(f"Lỗi truy vấn fallback: {e}")

        if len(final_syndromes) > 1:
            symptoms_list = []
            if user_symptoms: symptoms_list.append(user_symptoms)
            if raw_vision_data and isinstance(raw_vision_data, dict):
                symptoms_list.extend(raw_vision_data.get("detected_symptoms", []))
            if symptoms_list:
                final_syndromes = self._filter_syndromes_with_llm(symptoms_list, final_syndromes)

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
