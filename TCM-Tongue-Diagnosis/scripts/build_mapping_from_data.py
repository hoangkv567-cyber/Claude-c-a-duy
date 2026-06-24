import pandas as pd
import json
from collections import defaultdict

import yaml

# Đọc cấu hình
with open("config/config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)
csv_path = config.get("dataset", {}).get("csv_path", "data/Medicine_clean.csv")

# Đọc dataset
df = pd.read_csv(csv_path, encoding='utf-8')

# Khởi tạo mapping
mapping = defaultdict(set)

# Duyệt qua từng dòng
for _, row in df.iterrows():
    syndrome = row['hội_chứng']
    symptoms = [s.strip() for s in row['triệu_chứng'].split(',')] if pd.notna(row['triệu_chứng']) else []
    for symptom in symptoms:
        mapping[symptom].add(syndrome)

# Chuyển set thành list
mapping = {k: list(v) for k, v in mapping.items()}

# Lưu ra file JSON
with open("data/mapping/symptom_to_syndrome.json", 'w', encoding='utf-8') as f:
    json.dump(mapping, f, ensure_ascii=False, indent=2)

print(f"Da tao mapping cho {len(mapping)} trieu chung!")
