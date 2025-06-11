import simpy
import random
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt



# 🔧 1. Mevsime bağlı süre çarpanları (Yazın daha uzun olabilir)
season_multipliers = {
    'yaz': 1.3,
    'kış': 1.2,
    'bahar': 1.0,
    'sonbahar': 1.1
}

# 🏘️ 2. İzmir ilçelerinin altyapı özellikleri
districts_template = {
    'Konak':     {'infrastructure_score': 0.8, 'has_critical_infra': True,  'has_backup_power': True},
    'Bornova':   {'infrastructure_score': 0.7, 'has_critical_infra': True,  'has_backup_power': False},
    'Karşıyaka': {'infrastructure_score': 0.6, 'has_critical_infra': False, 'has_backup_power': False},
    'Aliağa':    {'infrastructure_score': 0.9, 'has_critical_infra': True,  'has_backup_power': True},
    'Tire':      {'infrastructure_score': 0.4, 'has_critical_infra': False, 'has_backup_power': False},
    'Çeşme':     {'infrastructure_score': 0.5, 'has_critical_infra': False, 'has_backup_power': True},
}

# 🔢 3. İlçeye öncelik ve müdahale süresi hesaplayan yardımcı fonksiyonlar
def compute_priority(d):
    priority = d['infrastructure_score']
    if d['has_critical_infra']:
        priority += 0.2
    return min(priority, 1.0)

def compute_recovery_time(d, season):
    base_time = 60
    if d['infrastructure_score'] < 0.5:
        base_time += 30
    if d['has_backup_power']:
        base_time *= 0.6
    base_time *= season_multipliers[season]
    return int(base_time)

# 🔁 4. Tek bir kesinti senaryosu oluşturur
def run_single_simulation(season):
    districts = {k: v.copy() for k, v in districts_template.items()}
    for d in districts.values():
        d['priority'] = compute_priority(d)
        d['recovery_time'] = compute_recovery_time(d, season)
    results = []

    def power_outage(env, name, recovery_time, priority, grid):
        start = env.now
        with grid.request(priority=1.0 - priority) as req:
            yield req
            yield env.timeout(recovery_time)
            end = env.now
            results.append({
                'ilçe': name,
                'altyapı_skoru': districts[name]['infrastructure_score'],
                'kritik_tesis': int(districts[name]['has_critical_infra']),
                'yedek_güç': int(districts[name]['has_backup_power']),
                'mevsim': season,
                'süre': end - start
            })

    def delayed_outage(env, delay, name, recovery_time, priority, grid):
        yield env.timeout(delay)
        yield env.process(power_outage(env, name, recovery_time, priority, grid))

    env = simpy.Environment()
    grid = simpy.PriorityResource(env, capacity=2)
    for name, info in districts.items():
        delay = random.randint(0, 20)
        env.process(delayed_outage(env, delay, name, info['recovery_time'], info['priority'], grid))
    env.run()
    return results

# 🔄 5. 100 farklı simülasyon çalıştırarak veri üret (veri seti oluşturuluyor)
print("🔄 100 farklı senaryo ile veri oluşturuluyor...")
all_data = []
for i in range(100):
    season = random.choice(list(season_multipliers.keys()))
    sim_data = run_single_simulation(season)
    all_data.extend(sim_data)

df = pd.DataFrame(all_data)

# 💾 Dosyaya yaz ve varsa eski verilerle birleştir
try:
    eski = pd.read_csv("genis_egitim_verisi.csv")
    birlesik = pd.concat([eski, df]).drop_duplicates()
except FileNotFoundError:
    birlesik = df

birlesik.to_csv("genis_egitim_verisi.csv", index=False)
print("✅ Veri seti 'genis_egitim_verisi.csv' olarak kaydedildi.")

# 🧠 6. XGBoost ile model eğitimi
print("🤖 XGBoost modeli eğitiliyor...")

X = df.drop(columns=["ilçe", "süre"])
y = df["süre"]

# 🧠 Mevsimi one-hot encode (model daha iyi öğrensin diye)
encoder = OneHotEncoder(sparse_output=False)
season_encoded = encoder.fit_transform(X[["mevsim"]])
season_df = pd.DataFrame(season_encoded, columns=encoder.get_feature_names_out(["mevsim"]))
X = pd.concat([X.drop(columns=["mevsim"]).reset_index(drop=True), season_df.reset_index(drop=True)], axis=1)

# 🔀 Veri setini eğitim/test olarak ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 📈 Modeli eğit
model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)



import matplotlib.pyplot as plt
import pandas as pd

# Özellik önem skorlarını al
importances = model.feature_importances_
feature_names = X.columns

# DataFrame oluştur
importance_df = pd.DataFrame({
    "Özellik": feature_names,
    "Önem": importances
}).sort_values(by="Önem", ascending=True)

# Grafik çizimi
plt.figure(figsize=(10, 6))
plt.barh(importance_df["Özellik"], importance_df["Önem"], color="seagreen")
plt.title("XGBoost Modelinde Özellik Önem Dereceleri")
plt.xlabel("Öneme Skoru")
plt.grid(True)
plt.show()


# 📊 7. Başarı ölçütleri
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"📊 Ortalama Hata (MAE): {mae:.2f} dakika")
print(f"📈 R² Skoru : {r2:.2f} ")

# 📉 8. Tahmin ve gerçek değerleri grafikle karşılaştır
print("📊 Gerçek ve tahmin edilen süreler görselleştiriliyor...")
compare_df = pd.DataFrame({"Gerçek Süre": y_test.values, "Tahmin Süre": y_pred})
compare_df.plot(kind='scatter', x="Gerçek Süre", y="Tahmin Süre", alpha=0.6)
plt.title("Gerçek vs Tahmin Edilen Süre")
plt.xlabel("Gerçek Süre (dk)")
plt.ylabel("Tahmin Edilen Süre (dk)")
plt.grid(True)
plt.tight_layout()
plt.show()

from joblib import dump

# Eğittiğin modeli ve encoder'ı kaydet
dump(model, "xgb_model.joblib")
dump(encoder, "onehot_encoder.joblib")

print("✅ Model ve encoder dosya olarak kaydedildi.")

