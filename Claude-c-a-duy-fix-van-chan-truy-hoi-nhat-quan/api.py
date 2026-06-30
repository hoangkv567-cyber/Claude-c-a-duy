# api.py
from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import shutil
import os
import logging

from src.fusion_pipeline import TCMFusionPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="HealthWatch AI Clinic")

# Cấu hình CORS để Web Frontend có thể gọi được API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Trong thực tế nên để domain cụ thể
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Khởi tạo bộ não AI
logger.info("Đang khởi động AI Backend...")
fusion_engine = TCMFusionPipeline()
os.makedirs("temp_uploads", exist_ok=True) # Thư mục lưu ảnh tạm
logger.info("Sẵn sàng!")

@app.post("/api/diagnose")
async def diagnose(
    symptoms: str = Form(""), 
    face_img: UploadFile = File(None), 
    tongue_img: UploadFile = File(None)
):
    # Làm sạch dữ liệu rác truyền từ frontend (nếu JS truyền biến undefined/null ở dạng chuỗi)
    if symptoms:
        s_val = symptoms.strip().lower()
        if s_val in ["undefined", "null", "none"]:
            symptoms = ""

    # Kiểm tra bắt buộc: Phải cung cấp ít nhất triệu chứng bằng văn bản HOẶC tải lên ít nhất một hình ảnh
    if not symptoms.strip() and not face_img and not tongue_img:
        raise HTTPException(
            status_code=400, 
            detail="Bắt buộc phải cung cấp triệu chứng lâm sàng bằng văn bản hoặc tải lên ít nhất một hình ảnh (ảnh sắc mặt hoặc ảnh lưỡi)."
        )

    import uuid
    import tempfile
    face_path, tongue_path = None, None
    try:
        # Lưu file tạm với tên độc nhất trong thư mục temp của hệ điều hành để tránh lỗi quyền ghi
        temp_dir = tempfile.gettempdir()
        if face_img:
            ext = os.path.splitext(face_img.filename)[1] or ".jpg"
            face_path = os.path.join(temp_dir, f"{uuid.uuid4()}{ext}")
            with open(face_path, "wb") as buffer:
                shutil.copyfileobj(face_img.file, buffer)
        if tongue_img:
            ext = os.path.splitext(tongue_img.filename)[1] or ".jpg"
            tongue_path = os.path.join(temp_dir, f"{uuid.uuid4()}{ext}")
            with open(tongue_path, "wb") as buffer:
                shutil.copyfileobj(tongue_img.file, buffer)

        # Gọi hệ thống hợp nhất chẩn đoán
        result = fusion_engine.run_diagnosis(
            user_symptoms=symptoms, 
            face_img_path=face_path, 
            tongue_img_path=tongue_path
        )
        return {"status": "success", "data": result}
    except Exception as e:
        logger.error(f"Lỗi: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Xóa các file tạm sau khi đã xử lý xong
        for path in [face_path, tongue_path]:
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                    logger.info(f"Đã dọn dẹp file tạm: {path}")
                except Exception as e:
                    logger.warning(f"Không thể xóa file tạm {path}: {e}")

from pydantic import BaseModel

class SymptomsRequest(BaseModel):
    symptoms: str

@app.post("/api/related-symptoms")
async def get_related_symptoms_endpoint(req: SymptomsRequest):
    symptoms_text = req.symptoms
    if not symptoms_text.strip():
        return {"status": "success", "data": []}
        
    try:
        # 1. Trích xuất từ khóa triệu chứng
        terms = fusion_engine.qa_pipeline._preprocess_question(symptoms_text)
        if not terms:
            return {"status": "success", "data": []}
            
        # 2. Tìm tên triệu chứng chính xác trong DB
        matched_db_names = []
        with fusion_engine.qa_pipeline.driver.session() as session:
            for term in terms:
                pattern = fusion_engine.qa_pipeline._word_boundary_pattern(term)
                q_match = """
                MATCH (t:TrieuChung)
                WHERE toLower(t.name) =~ $pattern
                RETURN t.name AS name
                """
                for rec in session.run(q_match, pattern=pattern):
                    matched_db_names.append(rec["name"].lower())
                    
        if not matched_db_names:
            return {"status": "success", "data": []}
            
        # 3. Tìm các triệu chứng liên quan đồng xuất hiện trong cùng Hội chứng (HoiChung)
        cypher = """
        MATCH (h:HoiChung)-[:CÓ_BIỂU_HIỆN]->(t:TrieuChung)
        WHERE toLower(t.name) IN $matched_names
        MATCH (h)-[:CÓ_BIỂU_HIỆN]->(t_other:TrieuChung)
        WHERE NOT toLower(t_other.name) IN $matched_names
        RETURN t_other.name AS symptom, count(distinct h) AS frequency
        ORDER BY frequency DESC
        LIMIT 50
        """
        
        related_symptoms = []
        with fusion_engine.qa_pipeline.driver.session() as session:
            for rec in session.run(cypher, matched_names=matched_db_names):
                related_symptoms.append(rec["symptom"])
                
        return {"status": "success", "data": related_symptoms}
    except Exception as e:
        logger.error(f"Lỗi gợi ý triệu chứng: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
