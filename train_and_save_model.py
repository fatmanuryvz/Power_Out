import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor
from joblib import dump

# 📂 CSV dosyasından veri oku
df = pd.read_csv("genis_egitim_verisi.csv")

# 🎯 Özellik ve hedef değişkeni ayır
X = df.drop(columns=["ilçe", "süre"])
y = df["süre"]

# 🧠 Mevsimi one-hot encode et
encoder = OneHotEncoder(sparse_output=False)
season_encoded = encoder.fit_transform(X[["mevsim"]])
season_df = pd.DataFrame(season_encoded, columns=encoder.get_feature_names_out(["mevsim"]))
X = pd.concat([X.drop(columns=["mevsim"]).reset_index(drop=True), season_df.reset_index(drop=True)], axis=1)

# 🚀 Modeli eğit
model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
model.fit(X, y)

# 💾 Model ve encoder'ı dosyaya kaydet
dump(model, "xgb_model.joblib")
dump(encoder, "onehot_encoder.joblib")

print("✅ Model ve encoder başarıyla kaydedildi.")
