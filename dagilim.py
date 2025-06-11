import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import networkx as nx
import time

# 📌 1. CSV verisini oku, model eğit
df = pd.read_csv("genis_egitim_verisi.csv").drop_duplicates()
le = LabelEncoder()
df["mevsim_encoded"] = le.fit_transform(df["mevsim"])

X = df[["altyapı_skoru", "kritik_tesis", "yedek_güç", "mevsim_encoded"]]
y = df["süre"]

model = xgb.XGBRegressor()
model.fit(X, y)

# 📌 2. Tahmin yapılacak ilçeler
ilceler = {
    "Karşıyaka": [0.6, 0, 0, "yaz"],
    "Tire": [0.4, 0, 0, "yaz"],
    "Konak": [0.8, 1, 1, "yaz"],
    "Çeşme": [0.5, 0, 1, "yaz"]
}

mevsim_map = {"yaz": le.transform(["yaz"])[0]}
prediction_data = []

for ilce, row in ilceler.items():
    row[-1] = mevsim_map[row[-1]]
    input_df = pd.DataFrame([row], columns=X.columns)
    sure = model.predict(input_df)[0]
    prediction_data.append((ilce, round(sure, 1)))  # ⬅️ Süreyi 1 basamakla yuvarla

# 📌 Süreye göre artan sırada sıralama
prediction_data.sort(key=lambda x: x[1])

# 🔎 Gösterim
print("\n🔵 Müdahale Öncelik Sırası (AI Tahmini):")
for i, (ilce, sure) in enumerate(prediction_data, 1):
    print(f"{i}. {ilce:<10} - {sure} dk")

# 📌 3. İlçe bağlantıları (mantıksal yayılım)
edges = [
    ("Çeşme", "Konak"),
    ("Konak", "Karşıyaka"),
    ("Karşıyaka", "Tire")
]

G = nx.Graph()
G.add_edges_from(edges)
pos = nx.spring_layout(G, seed=42)

status = {ilce: 'red' for ilce, _ in prediction_data}  # Hepsi başlangıçta kırmızı

# 📊 4. Canlı animasyon
plt.ion()
fig, ax = plt.subplots(figsize=(8, 6))

for step, (ilce, sure) in enumerate(prediction_data, 1):
    status[ilce] = 'green'

    ax.clear()
    nx.draw(
        G, pos, with_labels=True,
        node_color=[status[n] for n in G.nodes()],
        node_size=2000, font_size=13, font_weight='bold', ax=ax,
        edge_color='gray', linewidths=2
    )

    # Başlık ve açıklama
    ax.set_title(f"🛠️ Müdahale Adımı {step}: {ilce} ({sure:.1f} dk)", fontsize=14, pad=20)
    ax.text(0.5, 1.1,
            "✔ AI destekli sistem, kısa sürede müdahale edilmesi gereken bölgeleri tahmin eder.",
            fontsize=11, ha='center', transform=ax.transAxes)

    # 🔘 Legend sağ alt köşeye, kırmızı & yeşil net görünür şekilde
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='white', label='Kesinti Devam Ediyor',
                   markerfacecolor='red', markersize=12),
        plt.Line2D([0], [0], marker='o', color='white', label='Müdahale Edildi',
                   markerfacecolor='green', markersize=12)
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10, frameon=True)

    plt.pause(3.0)  # ⏱️ Daha yavaş gösterim

plt.ioff()
plt.show()
