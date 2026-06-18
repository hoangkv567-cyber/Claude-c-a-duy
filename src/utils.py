import logging
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

def setup_logging(level=logging.INFO):
    """Cấu hình logging cho toàn bộ ứng dụng"""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)

def load_json_config(file_path: str) -> Dict[str, Any]:
    """Đọc file cấu hình JSON"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"Lỗi khi đọc file cấu hình {file_path}: {e}")
        return {}

def save_json_result(file_path: str, data: Any, ensure_ascii: bool = False):
    """Lưu kết quả ra file JSON"""
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=ensure_ascii, indent=2)
        logging.info(f"Đã lưu kết quả vào {file_path}")
    except Exception as e:
        logging.error(f"Lỗi khi lưu file {file_path}: {e}")

def format_response(data: Dict) -> Dict:
    """Format dữ liệu trả về cho API"""
    return {
        "status": "success" if "error" not in data else "error",
        "data": data
    }

logger = setup_logging()

def normalize_symptoms_text(text: str) -> str:
    if not text:
        return text
    import re
    # nôn -> nôn mửa (chỉ thay thế nếu không được theo sau bởi 'mửa')
    text = re.sub(r'\bnôn\b(?!\s+mửa)', 'nôn mửa', text, flags=re.IGNORECASE)
    # mệt -> mệt mỏi (chỉ thay thế nếu không được theo sau bởi 'mỏi')
    text = re.sub(r'\bmệt\b(?!\s+mỏi)', 'mệt mỏi', text, flags=re.IGNORECASE)
    # khát -> khát nước (chỉ thay thế nếu không được theo sau bởi 'nước')
    text = re.sub(r'\bkhát\b(?!\s+nước)', 'khát nước', text, flags=re.IGNORECASE)
    # sốt -> phát sốt (chỉ thay thế nếu không đi sau 'phát')
    text = re.sub(r'(?<!phát\s)\bsốt\b', 'phát sốt', text, flags=re.IGNORECASE)
    # trướng -> bụng chướng (chỉ thay thế nếu không đi sau 'bụng')
    text = re.sub(r'(?<!bụng\s)\btrướng\b', 'bụng chướng', text, flags=re.IGNORECASE)
    # chướng -> bụng chướng (chỉ thay thế nếu không đi sau 'bụng')
    text = re.sub(r'(?<!bụng\s)\bchướng\b', 'bụng chướng', text, flags=re.IGNORECASE)
    return text

