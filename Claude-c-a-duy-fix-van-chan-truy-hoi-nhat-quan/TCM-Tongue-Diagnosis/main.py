import sys
import json
import argparse
import os
from src.pipeline import TCMTonguePipeline
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

    parser = argparse.ArgumentParser(description="Chẩn đoán Đông y qua ảnh (Lưỡi & Mặt)")
    
    # Đổi cả hai thành tham số có cờ (--)
    parser.add_argument("--tongue_image", help="Đường dẫn đến file ảnh lưỡi")
    parser.add_argument("--face_image", help="Đường dẫn đến file ảnh mặt")
    parser.add_argument("--config", default="config/config.yaml", help="File cấu hình")
    parser.add_argument("--output", help="File output JSON (mặc định in ra console)")
    args = parser.parse_args()

    # Kiểm tra: Bắt buộc phải có ít nhất 1 trong 2 ảnh
    if not args.tongue_image and not args.face_image:
        print("Lỗi: Hệ thống cần dữ liệu đầu vào. Vui lòng cung cấp --tongue_image hoặc --face_image (hoặc cả hai).")
        sys.exit(1)

    # Kiểm tra file ảnh lưỡi nếu được cung cấp
    if args.tongue_image and not os.path.exists(args.tongue_image):
        print(f"Lỗi: File ảnh lưỡi '{args.tongue_image}' không tồn tại!")
        sys.exit(1)

    # Kiểm tra file ảnh mặt nếu được cung cấp
    if args.face_image and not os.path.exists(args.face_image):
        print(f"Lỗi: File ảnh mặt '{args.face_image}' không tồn tại!")
        sys.exit(1)

    # Load config
    config = load_config(args.config)
    
    # Khởi tạo và chạy pipeline phù hợp với đầu vào
    if args.tongue_image and args.face_image:
        # Nếu có cả 2 ảnh, chạy kết hợp lưỡi & sắc mặt
        pipeline = TCMTongueFacePipeline(config=config)
        # Chỉ định rõ tên tham số để không bị lộn vòng
        result = pipeline.run(tongue_image_path=args.tongue_image, face_image_path=args.face_image)
    elif args.tongue_image:
        # Chỉ có ảnh lưỡi
        pipeline = TCMTonguePipeline(config=config, modality="tongue")
        result = pipeline.run(tongue_image_path=args.tongue_image)
    else:
        # Chỉ có ảnh mặt
        pipeline = TCMTonguePipeline(config=config, modality="face")
        # Điểm mấu chốt: Phải gọi đúng face_image_path
        result = pipeline.run(face_image_path=args.face_image)
    
    # Xuất kết quả
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"Đã lưu kết quả vào {args.output}")
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    
    pipeline.close()

if __name__ == "__main__":
    main()