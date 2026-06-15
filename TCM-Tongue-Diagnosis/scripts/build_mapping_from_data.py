import pandas as pd
import json
from collections import defaultdict

# Đọc dataset
df = pd.read_csv("data/tcm_data_600_clean.csv", encoding='utf-8')

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
