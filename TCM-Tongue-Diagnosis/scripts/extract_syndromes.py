import yaml

with open("config/config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)
csv_path = config.get("dataset", {}).get("csv_path", "data/Medicine_clean.csv")

df = pd.read_csv(csv_path, encoding='utf-8')
unique_syndromes = df['hội_chứng'].dropna().unique()
print(f"Tổng số hội chứng duy nhất: {len(unique_syndromes)}")
for s in sorted(unique_syndromes):
    print(f"- {s}")
