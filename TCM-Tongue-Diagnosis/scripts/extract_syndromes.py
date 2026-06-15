import pandas as pd

df = pd.read_csv("data/tcm_data_600_clean.csv", encoding='utf-8')
unique_syndromes = df['hội_chứng'].dropna().unique()
print(f"Tổng số hội chứng duy nhất: {len(unique_syndromes)}")
for s in sorted(unique_syndromes):
    print(f"- {s}")
