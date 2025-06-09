import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Judul Aplikasi
st.title("📈 Prediksi Inflasi Bulanan Indonesia dengan LSTM")

# Load Dataset
st.cache_data
def load_data():
    df = pd.read_excel("Data Inflasi (3).xlsx")
    df["Bulan"] = pd.date_range(start="2003-01-01", periods=len(df), freq="M")
    df.set_index("Bulan", inplace=True)
    return df

df = load_data()

# Tampilkan data asli
st.subheader("📊 Data Inflasi Asli")
st.line_chart(df["Data_Inflasi"])

# Normalisasi
scaler = MinMaxScaler()
df["Scaled"] = scaler.fit_transform(df[["Data_Inflasi"]])

# Load model
model = load_model("model_inflasi.h5")
seq_len = 12

# Buat data historis untuk evaluasi model
X_pred = []
for i in range(len(df) - seq_len):
    X_pred.append(df["Scaled"].values[i:i+seq_len])
X_pred = np.array(X_pred).reshape(-1, seq_len, 1)

# Prediksi historis
predictions = model.predict(X_pred)
df_pred = df.iloc[seq_len:].copy()
df_pred["Prediksi_Inflasi"] = scaler.inverse_transform(predictions)

# Tampilkan plot aktual vs prediksi
st.subheader("📉 Prediksi vs Aktual")
fig, ax = plt.subplots()
ax.plot(df_pred.index, df_pred["Data_Inflasi"], label="Aktual")
ax.plot(df_pred.index, df_pred["Prediksi_Inflasi"], label="Prediksi")
ax.set_xlabel("Waktu")
ax.set_ylabel("Inflasi (%)")
ax.legend()
st.pyplot(fig)

# Input user: berapa bulan ke depan ingin diprediksi
st.subheader("🔮 Prediksi Inflasi Bulan Mendatang")
n_months = st.number_input("Masukkan jumlah bulan ke depan untuk diprediksi:", min_value=1, max_value=36, value=12)

# Prediksi masa depan (berantai)
last_seq = df["Scaled"].values[-seq_len:].tolist()
future_preds_scaled = []

for _ in range(n_months):
    input_seq = np.array(last_seq[-seq_len:]).reshape(1, seq_len, 1)
    pred_scaled = model.predict(input_seq)
    future_preds_scaled.append(pred_scaled[0, 0])
    last_seq.append(pred_scaled[0, 0])

# Kembalikan ke skala asli
future_preds = scaler.inverse_transform(np.array(future_preds_scaled).reshape(-1, 1))

# Buat tanggal prediksi
last_date = df.index[-1]
future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=n_months, freq='M')

# Buat DataFrame hasil prediksi
future_df = pd.DataFrame({
    "Bulan": future_dates.strftime("%Y-%m"),
    "Prediksi_Inflasi (%)": future_preds.flatten()
})

# Tampilkan tabel dan grafik
st.dataframe(future_df)

fig2, ax2 = plt.subplots()
ax2.plot(future_dates, future_preds, marker='o')
ax2.set_title(f"Prediksi Inflasi {n_months} Bulan ke Depan")
ax2.set_xlabel("Bulan")
ax2.set_ylabel("Inflasi (%)")
ax2.grid(True)
st.pyplot(fig2)
