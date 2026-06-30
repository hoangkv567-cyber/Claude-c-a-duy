import sys
import json
import argparse
import os
from src.pipeline_dual import TCMTongueFacePipeline
from src.config_loader import load_config

def main():
    # Reconfigure stdout to use utf-8 to handle Unicode characters on Windows terminal
    if sys.stdout.encoding != 'utf-8':
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except AttributeError:
            import io
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    parser = argparse.ArgumentParser(description="Chẩn đoán lưỡi và mặt Đông y")
    parser.add_argument("tongue_image", help="Đường dẫn ảnh lưỡi")
    parser.add_argument("face_image", help="Đường dẫn ảnh mặt")
    parser.add_argument("--config", default="config/config.yaml", help="File cấu hình")
    parser.add_argument("--output", help="File output JSON")
    args = parser.parse_args()

    # Kiểm tra file ảnh tồn tại
    for path in [args.tongue_image, args.face_image]:
        if not os.path.exists(path):
            print(f"Lỗi: File ảnh '{path}' không tồn tại!")
            sys.exit(1)

    config = load_config(args.config)
    pipeline = TCMTongueFacePipeline(config=config)
    result = pipeline.run(args.tongue_image, args.face_image)
    
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"Đã lưu kết quả vào {args.output}")
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    
    pipeline.close()

if __name__ == "__main__":
    main()
