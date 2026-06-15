# src/fusion_pipeline.py
import logging
from src.pipeline import TCMTonguePipeline
from src.qa_system import TCMQA

logger = logging.getLogger(__name__)

class TCMFusionPipeline:
    def __init__(self, config: dict = None):
        """Khởi tạo toàn bộ lõi AI của hệ thống"""
        logger.info("Đang khởi tạo Hệ thống Hợp nhất (Fusion Pipeline)...")
        # Module Vọng chẩn (LLaVA - Phân tích ảnh)
        self.vision_pipeline = TCMTonguePipeline(config=config)
        
        # Module Vấn chẩn & Đồ thị (Qwen + Neo4j)
        self.qa_pipeline = TCMQA(config=config)
        
        # Gán neo4j_client cho qa_pipeline để tương thích ngược với code yêu cầu
        self.qa_pipeline.neo4j_client = self.vision_pipeline.neo4j_client
        logger.info("Khởi tạo hoàn tất!")

    def _extract_syndromes_from_text(self, text: str) -> list:
        """Trích xuất hội chứng từ câu mô tả của bệnh nhân bằng cách gọi QA pipeline."""
        if not text:
            return []
        try:
            # Dùng hàm execute_and_answer của QA pipeline để truy vấn
            # Ở đây ta tạo một câu hỏi đơn giản để lấy hội chứng
            result = self.qa_pipeline.execute_and_answer(f"{text}. Đó là hội chứng gì?")
            
            # Kiểm tra nếu trả về có key 'syndromes' trực tiếp
            if result and "syndromes" in result:
                return result.get("syndromes", [])
                
            # Trích xuất hội chứng từ key 'data' nếu có
            syndromes = []
            if result and "data" in result:
                for record in result["data"]:
                    for k, v in record.items():
                        if ("h.name" in k or "hoichung" in k.lower() or "syndrome" in k.lower()) and v:
                            syndromes.append(v)
            return list(set(syndromes))
        except Exception as e:
            logger.error(f"Lỗi khi trích xuất hội chứng từ văn bản: {e}")
            return []

    def _get_face_tongue_symptoms_for_syndrome(self, syndrome: str) -> dict:
        """Lấy các triệu chứng mặt và lưỡi dự kiến cho một hội chứng nhất định."""
        # mapping nghịch đảo từ hội chứng -> triệu chứng lưỡi/mặt xây dựng từ file mapping đã có
        tongue_symptoms = []
        face_symptoms = []
        
        try:
            import json
            import os
            
            # Đọc mapping lưỡi từ symptom_to_syndrome.json
            tongue_file = "data/mapping/symptom_to_syndrome.json"
            if os.path.exists(tongue_file):
                with open(tongue_file, "r", encoding="utf-8") as f:
                    tongue_map = json.load(f)
                for symptom, syndromes in tongue_map.items():
                    if syndrome in syndromes or any(syndrome.lower() == s.lower() for s in syndromes):
                        if symptom not in tongue_symptoms:
                            tongue_symptoms.append(symptom)
                            
            # Đọc mapping mặt từ face_to_syndrome.json
            face_file = "data/mapping/face_to_syndrome.json"
            if os.path.exists(face_file):
                with open(face_file, "r", encoding="utf-8") as f:
                    face_map = json.load(f)
                for symptom, syndromes in face_map.items():
                    if syndrome in syndromes or any(syndrome.lower() == s.lower() for s in syndromes):
                        if symptom not in face_symptoms:
                            face_symptoms.append(symptom)
        except Exception as e:
            logger.error(f"Lỗi khi đọc file mapping nghịch đảo: {e}")
            
        # Nếu không có từ file mapping, truy vấn nghịch đảo qua Neo4j
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
                logger.error(f"Lỗi khi lấy triệu chứng từ Neo4j: {e}")
                
        # Fallback mặc định nếu không có dữ liệu
        if not tongue_symptoms:
            tongue_symptoms = ["Lưỡi nhợt", "Lưỡi bệu có dấu răng"]
        if not face_symptoms:
            face_symptoms = ["Mặt nhợt nhạt"]
            
        return {
            "tongue": tongue_symptoms,
            "face": face_symptoms
        }

    def _filter_syndromes_with_llm(self, symptoms: list, candidate_syndromes: list) -> list:
        """Sử dụng LLM (Qwen) để lọc và chọn ra các hội chứng phù hợp nhất với tổ hợp triệu chứng, loại bỏ các hội chứng mâu thuẫn hoặc không liên quan."""
        if not candidate_syndromes:
            return []
        if len(candidate_syndromes) <= 1:
            return candidate_syndromes
            
        symptoms_str = ", ".join(symptoms)
        candidates_str = ", ".join(candidate_syndromes)
        
        prompt = f"""
        Vai trò: Bạn là một Bác sĩ Đông y giàu kinh nghiệm.
        Bệnh nhân có tổ hợp triệu chứng sau: {symptoms_str}
        Các hội chứng dự kiến được ánh xạ từ triệu chứng là: {candidates_str}
        
        Nhiệm vụ:
        1. Phân tích xem các hội chứng dự kiến trên có thực sự phù hợp với tổ hợp triệu chứng của bệnh nhân hay không.
        2. Loại bỏ các hội chứng mâu thuẫn nhau về mặt lâm sàng (ví dụ: Phong hàn là ngoại cảm cấp tính thường không đi kèm với các dấu hiệu nội nhân mãn tính/âm hư như lưỡi có vết nứt, trừ khi có bệnh cảnh đặc biệt phức tạp).
        3. Chọn ra từ 1 đến tối đa 3 hội chứng chính xác nhất phản ánh đúng thực tế lâm sàng của tổ hợp triệu chứng.
        4. Trả về kết quả dưới dạng danh sách tên các hội chứng được chọn, phân cách bằng dấu phẩy. Không giải thích gì thêm.
        
        Ví dụ đầu ra:
        Âm hư, Tân dịch thương
        """
        try:
            import ollama
            response = ollama.chat(
                model=self.qa_pipeline.llm_model,
                messages=[
                    {"role": "system", "content": "Bạn là chuyên gia chẩn đoán Đông y. Chỉ trả về danh sách hội chứng được chọn, phân cách bằng dấu phẩy, không giải thích gì thêm."},
                    {"role": "user", "content": prompt}
                ],
                options={
                    "temperature": 0.0,
                    "seed": 42
                }
            )
            ans = response['message']['content'].strip()
            # Parse câu trả lời
            selected = [s.strip() for s in ans.replace("\n", ",").split(",") if s.strip()]
            # Chỉ giữ các hội chứng thực sự nằm trong danh sách candidates ban đầu để tránh LLM bịa đặt
            valid_selected = [s for s in selected if any(s.lower() == cand.lower() for cand in candidate_syndromes)]
            if valid_selected:
                # Trả về theo đúng chữ hoa thường của candidates ban đầu
                final = []
                for v in valid_selected:
                    for cand in candidate_syndromes:
                        if v.lower() == cand.lower() and cand not in final:
                            final.append(cand)
                return final
            return candidate_syndromes
        except Exception as e:
            logger.error(f"Lỗi khi lọc hội chứng bằng LLM: {e}")
            return candidate_syndromes

    def _generate_explainable_answer(self, user_symptoms: str, detected_symptoms: list, final_syndromes: list, kg_context: str) -> str:
        """Sử dụng LLM (Qwen) để sinh câu trả lời chẩn đoán tự nhiên, giải thích rõ ràng dựa trên biện chứng luận trị Đông y và trích dẫn trực tiếp đường đi trên Knowledge Graph."""
        detected_symptoms_str = ", ".join(detected_symptoms) if detected_symptoms else "Không có"
        syndromes_str = ", ".join(final_syndromes) if final_syndromes else "Không xác định rõ"
        
        prompt = f"""
        Bạn là một bác sĩ Y học Cổ truyền (Đông y) đầu ngành. Hãy chẩn đoán và giải thích chi tiết, thuyết phục cho bệnh nhân dựa trên phương pháp "Biện chứng luận trị".
        
        THÔNG TIN BỆNH NHÂN:
        - Triệu chứng bệnh nhân tự mô tả (Vấn chẩn): {user_symptoms if user_symptoms else "Không mô tả"}
        - Triệu chứng phát hiện qua phân tích hình ảnh khuôn mặt/lưỡi (Vọng chẩn): {detected_symptoms_str}
        - Hội chứng bệnh lý đã được xác định: {syndromes_str}
        
        DỮ LIỆU TRUY XUẤT TỪ ĐỒ THỊ TRI THỨC (KNOWLEDGE GRAPH):
        {kg_context}
        
        YÊU CẦU CỰC KỲ QUAN TRỌNG VỀ LOGIC VÀ ĐỒ THỊ (YÊU CẦU BẮT BUỘC):
        1. **Không suy diễn lệch pha**: Đối với mỗi hội chứng trong chẩn đoán, bạn CHỈ ĐƯỢC PHÉP liên kết và giải thích nó dựa trên các triệu chứng được ghi ở mục "- Triệu chứng CỦA BỆNH NHÂN khớp với hội chứng này". TUYỆT ĐỐI KHÔNG ĐƯỢC gán các triệu chứng khác của bệnh nhân cho hội chứng đó nếu chúng không khớp (ví dụ: Không được ghi "Triệu chứng [Mặt trắng nhợt] dẫn đến hội chứng [Âm hư]", hoặc "Triệu chứng [Lưỡi có vết nứt] dẫn đến hội chứng [Tâm tỳ lưỡng hư]").
        2. **Đường đi đồ thị chính xác**: Trích dẫn trực tiếp và chính xác đường đi trên đồ thị tri thức (Knowledge Graph paths). Tên quan hệ bài thuốc - vị thuốc bắt buộc phải là `BAO_GÔM` (không được ghi là BAO_GỒM hay BAO_GOM).
           Ví dụ minh họa cách viết đường đi:
           - "Triệu chứng [Tên triệu chứng] dẫn đến hội chứng [Tên hội chứng] thông qua quan hệ `(HoiChung)-[:CÓ_BIỂU_HIỆN]->(TrieuChung)`"
           - "Hội chứng [Tên hội chứng] thuộc nhóm bệnh [Tên bệnh lý] qua quan hệ `(BenhLy)-[:CHIA_THÀNH]->(HoiChung)`"
           - "Điều trị hội chứng này bằng bài thuốc [Tên bài thuốc] qua quan hệ `(HoiChung)-[:ĐƯỢC_ĐIỀU_TRỊ_BẰNG]->(BaiThuoc)`"
           - "Bài thuốc [Tên bài thuốc] gồm các vị thuốc như [Tên vị thuốc] qua quan hệ `(BaiThuoc)-[:BAO_GÔM]->(ViThuoc)`"
        
        CẤU TRÚC CHI TIẾT CÂU TRẢ LỜI:
        1. **Kết luận chẩn đoán**: Đưa ra kết luận rõ ràng về hội chứng bệnh lý của bệnh nhân.
        2. **Biện chứng luận trị (Giải thích lý do)**: Giải thích chi tiết và thuyết phục tại sao các triệu chứng khớp của bệnh nhân lại dẫn đến hội chứng chẩn đoán này theo lý luận Đông y (âm dương, khí huyết, tạng phủ, hàn nhiệt, hư thực).
        3. **Trích dẫn đường đi trên đồ thị tri thức (Knowledge Graph paths)**: Viết rõ các đường đi đồ thị cho từng hội chứng theo yêu cầu số 2 ở trên.
        4. **Phương dược điều trị**: Giới thiệu bài thuốc phù hợp và các vị thuốc tương ứng, giải thích sơ lược công dụng của bài thuốc/vị thuốc trong việc điều hòa cơ thể.
        5. **Lời khuyên**: Đưa ra lời khuyên sinh hoạt/dinh dưỡng phù hợp với thể trạng của bệnh nhân.
        
        Hãy trình bày bằng tiếng Việt, giọng điệu ân cần, chuyên nghiệp và khoa học. Tránh dùng các từ ngữ sáo rỗng.
        """
        try:
            import ollama
            response = ollama.chat(
                model=self.qa_pipeline.llm_model,
                messages=[
                    {"role": "system", "content": "Bạn là chuyên gia Đông y chẩn đoán chính xác tuyệt đối theo đồ thị tri thức, không suy diễn lệch pha, sử dụng quan hệ BAO_GÔM."},
                    {"role": "user", "content": prompt}
                ],
                options={
                    "temperature": 0.1,
                    "seed": 42
                }
            )
            return response['message']['content'].strip()
        except Exception as e:
            logger.error(f"Lỗi khi sinh câu trả lời giải thích bằng LLM: {e}")
            return f"Chẩn đoán hội chứng: {syndromes_str}. Chưa thể sinh câu trả lời chi tiết do lỗi hệ thống."

    def run_diagnosis(self, user_symptoms: str = "", face_img_path: str = None, tongue_img_path: str = None) -> dict:
        """
        Thực thi Tứ chẩn hợp tham: Kết hợp chữ viết và hình ảnh để chẩn đoán.
        """
        vision_analysis_text = ""
        raw_vision_data = None
        vision_syndromes = []

        # Bước 1: Xử lý Vọng chẩn (Nếu người dùng gửi ảnh)
        if face_img_path or tongue_img_path:
            logger.info("Bắt đầu phân tích hình ảnh qua LLaVA...")
            try:
                # Gọi hàm run của pipeline thị giác
                raw_vision_data = self.vision_pipeline.run(
                    tongue_image_path=tongue_img_path, 
                    face_image_path=face_img_path
                )
                
                # Trích xuất chuỗi mô tả từ kết quả trả về
                if raw_vision_data and isinstance(raw_vision_data, dict):
                    vision_analysis_text = raw_vision_data.get("analysis", "")
                elif isinstance(raw_vision_data, str):
                    vision_analysis_text = raw_vision_data
                else:
                    vision_analysis_text = str(raw_vision_data)
            except Exception as e:
                logger.error(f"Lỗi module Vision: {e}")

        if raw_vision_data and isinstance(raw_vision_data, dict):
            vision_syndromes = raw_vision_data.get("syndromes", raw_vision_data.get("possible_syndromes", []))
        else:
            vision_syndromes = []

        # Bước 2: Xử lý triệu chứng từ văn bản (Vấn chẩn)
        text_syndromes = self._extract_syndromes_from_text(user_symptoms) if user_symptoms else []

        # Bước 3: Tổng hợp Logic
        # 1. Nếu cả văn bản và ảnh đều có, lấy giao của các hội chứng
        final_syndromes = []
        if text_syndromes and vision_syndromes:
            final_syndromes = list(set(text_syndromes) & set(vision_syndromes))
        
        # 2. Nếu chỉ có văn bản, lấy hội chứng từ văn bản
        elif text_syndromes:
            final_syndromes = text_syndromes
        
        # 3. Nếu chỉ có ảnh, lấy hội chứng từ ảnh
        elif vision_syndromes:
            final_syndromes = vision_syndromes
        
        # 4. Nếu không có từ nguồn nào, kết thúc
        else:
            pass

        # Tổ hợp chuỗi triệu chứng gộp để tìm kiếm/fallback
        combined_query = user_symptoms.strip()
        if vision_analysis_text:
            if combined_query:
                combined_query = f"{combined_query}, {vision_analysis_text}"
            else:
                combined_query = vision_analysis_text

        # Fallback nếu không xác định được trực tiếp từ mapping: chạy QA RAG truy vấn đồ thị để lấy hội chứng
        if not final_syndromes and combined_query:
            logger.info(f"Không xác định được hội chứng trực tiếp, chạy truy vấn đồ thị qua QA để trích xuất hội chứng...")
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
                    logger.info(f"Trích xuất thành công {len(final_syndromes)} hội chứng từ đồ thị: {final_syndromes}")
            except Exception as e:
                logger.error(f"Lỗi khi chạy truy vấn fallback lấy hội chứng: {e}")

        # Lọc lâm sàng bằng LLM để loại bỏ hội chứng không phù hợp hoặc mâu thuẫn
        if len(final_syndromes) > 1:
            symptoms_list = []
            if user_symptoms:
                symptoms_list.append(user_symptoms)
            if raw_vision_data and isinstance(raw_vision_data, dict):
                detected = raw_vision_data.get("detected_symptoms", [])
                symptoms_list.extend(detected)
            
            if symptoms_list:
                logger.info(f"Đang lọc lâm sàng cho các hội chứng {final_syndromes} với triệu chứng {symptoms_list}...")
                final_syndromes = self._filter_syndromes_with_llm(symptoms_list, final_syndromes)

        # Bước 4: Tạo câu trả lời tự nhiên có giải thích (Explainable Generation)
        qa_result = None
        if final_syndromes:
            # 1. Đọc lại toàn bộ file mapping để phục vụ đối chiếu triệu chứng khớp
            import json
            import os
            all_mappings = {}
            for filename in ["symptom_to_syndrome.json", "face_to_syndrome.json"]:
                path = os.path.join("data", "mapping", filename)
                if os.path.exists(path):
                    try:
                        with open(path, "r", encoding="utf-8") as f:
                            all_mappings.update(json.load(f))
                    except Exception as e:
                        logger.error(f"Lỗi đọc {filename} để đối chiếu: {e}")

            # Danh sách toàn bộ triệu chứng thực tế của bệnh nhân
            patient_symptoms = []
            if user_symptoms:
                patient_symptoms.extend([s.strip() for s in user_symptoms.split(",") if s.strip()])
            if raw_vision_data and isinstance(raw_vision_data, dict):
                detected = raw_vision_data.get("detected_symptoms", [])
                patient_symptoms.extend(detected)
            patient_symptoms = list(set([s for s in patient_symptoms if s]))

            # 2. Truy xuất thông tin từ đồ thị tri thức (Neo4j) cho các hội chứng đã chọn
            kg_context_lines = []
            treatments = []
            
            for syndrome in final_syndromes:
                # Tìm các triệu chứng của bệnh nhân thực sự tương thích với hội chứng này
                matching_patient_symptoms = []
                for s in patient_symptoms:
                    mapped_syndromes = all_mappings.get(s, [])
                    if syndrome in mapped_syndromes or any(syndrome.lower() == ms.lower() for ms in mapped_syndromes):
                        matching_patient_symptoms.append(s)
                        continue
                    try:
                        db_symptoms = self.qa_pipeline.neo4j_client.get_symptoms_by_syndrome(syndrome)
                        if any(s.lower() in ds.lower() or ds.lower() in s.lower() for ds in db_symptoms):
                            matching_patient_symptoms.append(s)
                    except Exception as e:
                        logger.error(f"Lỗi kiểm tra Neo4j matching symptoms: {e}")
                
                matching_patient_symptoms = list(set(matching_patient_symptoms))
                
                try:
                    diseases = self.qa_pipeline.neo4j_client.get_diseases_by_syndrome(syndrome)
                    db_symptoms = self.qa_pipeline.neo4j_client.get_symptoms_by_syndrome(syndrome)
                    treatment = self.qa_pipeline.neo4j_client.get_treatment_by_syndrome(syndrome)
                    
                    diseases_str = ", ".join(diseases) if diseases else "Chưa rõ"
                    db_symptoms_str = ", ".join(db_symptoms) if db_symptoms else "Chưa rõ"
                    matching_str = ", ".join(matching_patient_symptoms) if matching_patient_symptoms else "Không có triệu chứng khớp trực tiếp (suy luận từ bệnh cảnh)"
                    
                    kg_context_lines.append(f"--- Hội chứng: {syndrome} ---")
                    kg_context_lines.append(f"- Triệu chứng CỦA BỆNH NHÂN khớp với hội chứng này: {matching_str}")
                    kg_context_lines.append(f"- Bệnh lý liên quan (Path: BenhLy -> CHIA_THÀNH -> HoiChung): {diseases_str}")
                    kg_context_lines.append(f"- Triệu chứng biểu hiện lý thuyết trong database (Path: HoiChung -> CÓ_BIỂU_HIỆN -> TrieuChung): {db_symptoms_str}")
                    
                    if treatment:
                        treatments.append(treatment)
                        bai_thuoc = treatment.get("bai_thuoc", "Chưa rõ")
                        vi_thuoc = ", ".join(treatment.get("vi_thuoc", [])) if treatment.get("vi_thuoc") else "Chưa rõ"
                        kg_context_lines.append(f"- Bài thuốc điều trị (Path: HoiChung -> ĐƯỢC_ĐIỀU_TRỊ_BẰNG -> BaiThuoc): {bai_thuoc}")
                        kg_context_lines.append(f"- Các vị thuốc trong bài (Path: BaiThuoc -> BAO_GÔM -> ViThuoc): {vi_thuoc}")
                    else:
                        kg_context_lines.append(f"- Bài thuốc điều trị: Chưa có bài thuốc phù hợp trong hệ thống.")
                except Exception as e:
                    logger.error(f"Lỗi khi truy xuất dữ liệu Neo4j cho hội chứng {syndrome}: {e}")

            kg_context_str = "\n".join(kg_context_lines)
            
            # 3. Sử dụng LLM để sinh câu trả lời giải thích biện chứng luận trị và chỉ ra đường đi đồ thị
            detected_symptoms = []
            if raw_vision_data and isinstance(raw_vision_data, dict):
                detected_symptoms = raw_vision_data.get("detected_symptoms", [])
                
            logger.info("Đang sinh câu trả lời giải thích biện chứng luận trị bằng LLM...")
            explainable_answer = self._generate_explainable_answer(
                user_symptoms=user_symptoms,
                detected_symptoms=detected_symptoms,
                final_syndromes=final_syndromes,
                kg_context=kg_context_str
            )
            
            qa_result = {
                "answer": explainable_answer,
                "data": treatments,
                "syndromes": final_syndromes
            }
        else:
            # Fallback nếu hoàn toàn không tìm thấy hội chứng
            qa_result = {
                "answer": "Hệ thống chưa tìm thấy bệnh lý hoặc hội chứng phù hợp với tổ hợp triệu chứng của bạn.",
                "data": []
            }

        # Bước 5: Đóng gói kết quả trả về
        return {
            "source": "Tứ chẩn hợp tham (Fusion)",
            "input_fusion": combined_query,
            "vision_details": raw_vision_data,
            "diagnosis_result": qa_result
        }


    def close(self):
        """Giải phóng tài nguyên khi tắt hệ thống"""
        if hasattr(self, 'qa_pipeline'):
            self.qa_pipeline.close()
        # Nếu TCMTonguePipeline có hàm close, hãy gọi nó ở đây
        if hasattr(self, 'vision_pipeline') and hasattr(self.vision_pipeline, 'close'):
            self.vision_pipeline.close()
        logger.info("Đã giải phóng tài nguyên.")
