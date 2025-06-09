import streamlit as st
import pandas as pd
import numpy as np
import os
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Judul Aplikasi
st.title("📈 Prediksi Inflasi Bulanan Indonesia dengan LSTM")

# Fungsi load data dengan cache
@st.cache_data
def load_data():
    df = pd.read_excel("Data Inflasi (3).xlsx")
    df["Bulan"] = pd.date_range(start="2003-01-01", periods=len(df), freq="M")
    df.set_index("Bulan", inplace=True)
    return df

# Load dataset
df = load_data()

# Cek panjang data minimal
seq_len = 12
if len(df) <= seq_len:
    st.error(f"❌ Data terlalu sedikit. Minimal {seq_len + 1} baris diperlukan. Saat ini hanya {len(df)}.")
    st.stop()

# Tampilkan data asli
st.subheader("📊 Data Inflasi Asli")
st.line_chart(df["Data_Inflasi"])

# Normalisasi data
scaler = MinMaxScaler()
df["Scaled"] = scaler.fit_transform(df[["Data_Inflasi"]])

# Validasi file model
if not os.path.exists("model_inflasi.h5"):
    st.error("❌ File model_inflasi.h5 tidak ditemukan.")
    st.stop()

# Load model
model = load_model("model_inflasi.h5")

# Buat data historis untuk evaluasi
X_pred = []
for i in range(len(df) - seq_len):
    X_pred.append(df["Scaled"].values[i:i + seq_len])
X_pred = np.array(X_pred).reshape(-1, seq_len, 1)

# Prediksi historis
predictions = model.predict(X_pred)
df_pred = df.iloc[seq_len:].copy()
df_pred["Prediksi_Inflasi"] = scaler.inverse_transform(predictions)

# Evaluasi model
mse = mean_squared_error(df_pred["Data_Inflasi"], df_pred["Prediksi_Inflasi"])
mae = mean_absolute_error(df_pred["Data_Inflasi"], df_pred["Prediksi_Inflasi"])
r2 = r2_score(df_pred["Data_Inflasi"], df_pred["Prediksi_Inflasi"])

st.write("📊 **Evaluasi Model Terhadap Data Historis:**")
st.markdown(f"""
- ✅ MSE (Mean Squared Error): `{mse:.4f}`
- ✅ MAE (Mean Absolute Error): `{mae:.4f}`
- ✅ R² Score: `{r2:.4f}`
""")

# Plot prediksi vs aktual
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=df_pred.index, y=df_pred["Data_Inflasi"],
    mode='lines+markers',
    name='Aktual'
))

fig.add_trace(go.Scatter(
    x=df_pred.index, y=df_pred["Prediksi_Inflasi"],
    mode='lines+markers',
    name='Prediksi'
))

fig.update_layout(
    title="📉 Prediksi vs Aktual Inflasi",
    xaxis_title="Waktu",
    yaxis_title="Inflasi (%)",
    hovermode="x unified",
    height=500
)

st.plotly_chart(fig, use_container_width=True)

# Input jumlah bulan ke depan
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

# Buat tanggal-tanggal prediksi
last_date = df.index[-1]
future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=n_months, freq='M')

# Buat DataFrame hasil prediksi
future_df = pd.DataFrame({
    "Bulan": future_dates.strftime("%Y-%m"),
    "Prediksi_Inflasi (%)": future_preds.flatten()
})

# Tampilkan tabel dan grafik hasil prediksi
st.dataframe(future_df)

fig2, ax2 = plt.subplots()
ax2.plot(future_dates, future_preds, marker='o')
ax2.set_title(f"Prediksi Inflasi {n_months} Bulan ke Depan")
ax2.set_xlabel("Bulan")
ax2.set_ylabel("Inflasi (%)")
ax2.grid(True)
plt.xticks(rotation=45) 

fig2.autofmt_xdate()

st.pyplot(fig2)
