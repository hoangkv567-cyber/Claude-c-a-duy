from fastapi import FastAPI, UploadFile, File, HTTPException
import uvicorn
import tempfile
import os
from typing import Optional
from src.pipeline import TCMTonguePipeline
from src.pipeline_dual import TCMTongueFacePipeline
from src.config_loader import load_config
from src.utils import logger

# Load configuration
config = load_config("config/config.yaml")

app = FastAPI(
    title="TCM Tongue & Face Diagnosis API",
    description="API for analyzing tongue and face images using LLaVA (Ollama) and Neo4j."
)

# Global pipelines initialized on startup
pipeline_tongue = None
pipeline_face = None
pipeline_dual = None

@app.on_event("startup")
async def startup_event():
    global pipeline_tongue, pipeline_face, pipeline_dual
    logger.info("Initializing pipelines on startup...")
    pipeline_tongue = TCMTonguePipeline(config=config, modality="tongue")
    pipeline_face = TCMTonguePipeline(config=config, modality="face")
    pipeline_dual = TCMTongueFacePipeline(config=config)

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Closing pipeline connections on shutdown...")
    if pipeline_tongue:
        pipeline_tongue.close()
    if pipeline_face:
        pipeline_face.close()
    if pipeline_dual:
        pipeline_dual.close()

@app.post("/diagnose")
async def diagnose(
    image: Optional[UploadFile] = File(None),
    tongue_image: Optional[UploadFile] = File(None),
    face_image: Optional[UploadFile] = File(None),
    modality: Optional[str] = None
):
    """
    Perform tongue, face, or dual diagnosis.
    
    Inputs can be:
    - tongue_image AND face_image: Dual analysis
    - tongue_image ONLY: Tongue analysis
    - face_image ONLY: Face analysis
    - image ONLY: Single analysis based on `modality` (defaults to "tongue")
    """
    # Check what images were uploaded
    temp_files = []
    
    try:
        # 1. Handle dual image case explicitly
        if tongue_image and face_image:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_t:
                tmp_t.write(await tongue_image.read())
                tongue_path = tmp_t.name
                temp_files.append(tongue_path)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_f:
                tmp_f.write(await face_image.read())
                face_path = tmp_f.name
                temp_files.append(face_path)
            
            logger.info("Running dual diagnosis pipeline")
            result = pipeline_dual.run(tongue_path, face_path)
            return result

        # 2. Handle single inputs with explicit fields
        elif tongue_image:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_t:
                tmp_t.write(await tongue_image.read())
                tongue_path = tmp_t.name
                temp_files.append(tongue_path)
            
            logger.info("Running tongue diagnosis pipeline")
            result = pipeline_tongue.run(tongue_path)
            return result

        elif face_image:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_f:
                tmp_f.write(await face_image.read())
                face_path = tmp_f.name
                temp_files.append(face_path)
            
            logger.info("Running face diagnosis pipeline")
            result = pipeline_face.run(face_path)
            return result

        # 3. Handle backwards compatible 'image' field
        elif image:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_img:
                tmp_img.write(await image.read())
                img_path = tmp_img.name
                temp_files.append(img_path)
            
            # Decide modality based on query parameter
            mode = (modality or "tongue").lower().strip()
            if mode == "face":
                logger.info("Running face diagnosis pipeline (via 'image' field)")
                result = pipeline_face.run(img_path)
            elif mode == "tongue":
                logger.info("Running tongue diagnosis pipeline (via 'image' field)")
                result = pipeline_tongue.run(img_path)
            else:
                raise HTTPException(
                    status_code=400,
                    detail=f"Modality '{modality}' không hợp lệ cho trường 'image'. Chọn 'tongue' hoặc 'face'."
                )
            return result

        else:
            raise HTTPException(
                status_code=400,
                detail="Vui lòng cung cấp ít nhất một file ảnh (tongue_image, face_image, hoặc image)."
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Lỗi trong quá trình xử lý chẩn đoán: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Lỗi hệ thống: {str(e)}")

    finally:
        # Cleanup temporary files
        for path in temp_files:
            if os.path.exists(path):
                try:
                    os.remove(path)
                except Exception as e:
                    logger.warning(f"Không thể xoá file tạm {path}: {e}")

if __name__ == "__main__":
    host = config.get("api", {}).get("host", "0.0.0.0")
    port = config.get("api", {}).get("port", 8000)
    uvicorn.run("app:app", host=host, port=port, reload=config.get("api", {}).get("debug", False))
