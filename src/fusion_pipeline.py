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

        # Gán neo4j_client cho qa_pipeline
        self.qa_pipeline.neo4j_client = self.vision_pipeline.neo4j_client
        logger.info("Khởi tạo hoàn tất!")

    def run_diagnosis(self, user_symptoms: str = "", face_img_path: str = None, tongue_img_path: str = None) -> dict:
        """
        Thực thi Tứ chẩn hợp tham: Kết hợp chữ viết và hình ảnh để chẩn đoán.
        PHIÊN BẢN ĐƠN GIẢN: Chỉ dùng QA pipeline, không dùng LLM để suy luận.
        """
        combined_query = user_symptoms.strip()
        
        # Nếu có ảnh, gọi vision pipeline để lấy triệu chứng
        if face_img_path or tongue_img_path:
            logger.info("Bắt đầu phân tích hình ảnh qua LLaVA...")
            try:
                raw_vision_data = self.vision_pipeline.run(
                    tongue_image_path=tongue_img_path, 
                    face_image_path=face_img_path
                )
                if raw_vision_data and isinstance(raw_vision_data, dict):
                    vision_text = raw_vision_data.get("analysis", "")
                    if vision_text:
                        if combined_query:
                            combined_query = f"{combined_query}, {vision_text}"
                        else:
                            combined_query = vision_text
            except Exception as e:
                logger.error(f"Lỗi module Vision: {e}")

        # Chỉ dùng QA pipeline để truy vấn
        if combined_query:
            logger.info(f"Đang truy vấn Graph RAG với từ khóa: {combined_query}")
            qa_result = self.qa_pipeline.execute_and_answer(combined_query)
            return {
                "source": "Tứ chẩn hợp tham (Fusion - Đơn giản)",
                "input_fusion": combined_query,
                "diagnosis_result": qa_result
            }
        else:
            return {
                "source": "Tứ chẩn hợp tham (Fusion)",
                "input_fusion": "",
                "diagnosis_result": {
                    "answer": "Vui lòng nhập triệu chứng hoặc tải lên hình ảnh.",
                    "data": []
                }
            }

    def close(self):
        """Giải phóng tài nguyên khi tắt hệ thống"""
        if hasattr(self, 'qa_pipeline'):
            self.qa_pipeline.close()
        if hasattr(self, 'vision_pipeline') and hasattr(self.vision_pipeline, 'close'):
            self.vision_pipeline.close()
        logger.info("Đã giải phóng tài nguyên.")
