import json
import pandas as pd

texts = []

with open("CrimeReport.txt", "r", encoding="utf-8") as f:
    for line in f:
        try:
            data = json.loads(line)
            text_value = data.get("text", "").strip()
            if text_value:
                texts.append(text_value)
        except Exception:
            continue

df = pd.DataFrame({"text": texts})
df.to_csv("text/data/crimereport.csv", index=False)

print("✅ CSV created at text/data/crimereport.csv")
print(f"Total rows: {len(df)}")

