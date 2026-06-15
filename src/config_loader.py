import yaml
from pathlib import Path

def load_config(config_path="config/config.yaml"):
    """
    Load file cấu hình YAML
    
    Args:
        config_path (str): Đường dẫn đến file cấu hình
        
    Returns:
        dict: Dictionary chứa cấu hình
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file cấu hình '{config_path}'")
        print("Sử dụng cấu hình mặc định...")
        return {
            "ollama": {"model": "llava:7b"},
            "neo4j": {"uri": "neo4j://localhost:7687", "user": "neo4j", "password": "12345678"},
            "mapping": {"symptom_to_syndrome": "data/mapping/symptom_to_syndrome.json"},
            "api": {"host": "0.0.0.0", "port": 8000}
        }
    except Exception as e:
        print(f"Lỗi khi đọc file cấu hình: {e}")
        return {}
