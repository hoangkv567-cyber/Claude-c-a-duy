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
    # Kiểm tra bắt buộc nhập triệu chứng và tải lên ít nhất một hình ảnh
    if not symptoms.strip():
        raise HTTPException(
            status_code=400, 
            detail="Bắt buộc phải cung cấp triệu chứng lâm sàng bằng văn bản."
        )
    if not face_img and not tongue_img:
        raise HTTPException(
            status_code=400, 
            detail="Bắt buộc phải tải lên ít nhất một hình ảnh (ảnh sắc mặt hoặc ảnh lưỡi)."
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

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
