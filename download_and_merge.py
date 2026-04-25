import os
import requests
import zipfile
import pandas as pd

# Měsíce, které chceme stáhnout
months_2025 = [f"2025-{str(m).zfill(2)}" for m in range(1, 13)]
months_2026 = [f"2026-{str(m).zfill(2)}" for m in range(1, 4)]

months = months_2025 + months_2026

base_url = "https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/1h/"

# Vytvoříme složku
os.makedirs("tmp", exist_ok=True)

csv_files = []

for month in months:
    filename = f"BTCUSDT-1h-{month}.zip"
    url = base_url + filename
    zip_path = f"tmp/{filename}"

    print(f"Stahuji {filename}...")

    r = requests.get(url)
    if r.status_code != 200:
        print(f"❌ Soubor {filename} neexistuje nebo je blokovaný.")
        continue

    with open(zip_path, "wb") as f:
        f.write(r.content)

    # Rozbalíme ZIP
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall("tmp")
        csv_name = f"BTCUSDT-1h-{month}.csv"
        csv_files.append(f"tmp/{csv_name}")

print("✔️ Všechny dostupné soubory staženy.")

# Spojíme CSV
dfs = []
for csv in csv_files:
    df = pd.read_csv(csv, header=None)
    dfs.append(df)

merged = pd.concat(dfs, ignore_index=True)
merged.to_csv("BTCUSDT_1h.csv", index=False, header=False)

print("🔥 Hotovo! Soubor BTCUSDT_1h.csv vytvořen.")

