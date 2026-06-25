import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import joblib
import base64
import plotly.express as px 
import os

st.set_page_config(page_title="Sistem Peringatan Dini Nelayan", layout="wide")

# =========================
# LOGIN ADMIN
# =========================
if "login_status" not in st.session_state:
    st.session_state.login_status = False

def login():
    st.sidebar.subheader("Login Admin")
    username = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type="password")

    if st.sidebar.button("Login"):
        if username == "admin" and password == "123":  # nanti bisa kamu ubah
            st.session_state.login_status = True
            st.success("Login berhasil")
        else:
            st.error("Username atau password salah")

login()

# =========================
# INIT KEY UPLOADER
# =========================
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

# =========================
# UPLOAD DATA (ADMIN ONLY)
# =========================
DATA_PATH = "data_latest.csv"

if st.session_state.login_status:  # 🔥 INI KUNCINYA

    st.sidebar.subheader("Upload Data Terbaru")

    uploaded_file = st.sidebar.file_uploader(
        "Upload file CSV",
        type=["csv"],
        key=st.session_state.uploader_key
    )

    if uploaded_file is not None:
        if st.sidebar.button("Proses & Update Data"):
            with st.spinner("Memproses data..."):
                df_new = pd.read_csv(uploaded_file)
                df_new.to_csv(DATA_PATH, index=False)

            st.success("Data berhasil diupdate!")

            # reset uploader
            st.session_state.uploader_key += 1

            st.rerun()

# =========================
# FUNGSI BACA FILE GAMBAR
# =========================
def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# =========================
# LOAD MODEL & SCALER
# =========================
@st.cache_resource
def load_my_model():
    model = load_model("lstm_model.h5", compile=False)
    scaler = joblib.load("scaler.save")
    return model, scaler

model, scaler = load_my_model()

# =========================
# PREPROCESS DATA
# =========================
def preprocess_data(df):
    if {'Time(UTC/GMT)', 'Hsig(m)', 'WindSpeed(knots)'}.issubset(df.columns):
        df = df[['Time(UTC/GMT)', 'Hsig(m)', 'WindSpeed(knots)']].copy()
        df.columns = ['time', 'hsig', 'wind']
    elif {'time', 'hsig', 'wind'}.issubset(df.columns):
        df = df[['time', 'hsig', 'wind']].copy()
    else:
        raise ValueError(
            "Kolom file tidak sesuai. Pastikan ada kolom "
            "'Time(UTC/GMT)', 'Hsig(m)', 'WindSpeed(knots)' "
            "atau 'time', 'hsig', 'wind'."
        )

    df['time'] = pd.to_datetime(df['time'])
    df = df.sort_values('time').reset_index(drop=True)
    return df

# =========================
# KLASIFIKASI STATUS
# =========================
def classify(hsig, wind):
    if hsig > 1.25 or wind > 15:
        return "Bahaya"
    elif hsig > 0.5 or wind > 10:
        return "Waspada"
    else:
        return "Aman"

def show_status_box(status, title_text):
    if status == "Aman":
        st.success(f"{title_text}: {status}")
    elif status == "Waspada":
        st.warning(f"{title_text}: {status}")
    else:
        st.error(f"{title_text}: {status}")

def find_closest_data(df, selected_datetime):
    df_copy = df.copy()
    df_copy["selisih"] = (df_copy["time"] - selected_datetime).abs()
    nearest_row = df_copy.loc[df_copy["selisih"].idxmin()]
    return nearest_row

# =========================
# FUNGSI PREDIKSI LSTM
# =========================
def predict_lstm(df, selected_datetime, timestep=24):
    df_before = df[df["time"] < selected_datetime].copy()

    if len(df_before) < timestep:
        return None, None

    last_sequence = df_before.tail(timestep)[["hsig", "wind"]].values

    last_sequence_scaled = scaler.transform(last_sequence)

    X_input = last_sequence_scaled.reshape(1, timestep, 2)

    pred_scaled = model.predict(X_input)

    dummy = np.zeros((1, scaler.n_features_in_))
    dummy[0, 0] = pred_scaled[0, 0]
    dummy[0, 1] = pred_scaled[0, 1]

    inv = scaler.inverse_transform(dummy)

    pred_hsig = float(inv[0, 0])
    pred_wind = float(inv[0, 1])

    return pred_hsig, pred_wind

# =========================
# MULTI STEP FORECAST
# =========================
def forecast_future(df, selected_datetime, steps=24, timestep=8):
    df_before = df[df["time"] < selected_datetime].copy()

    if len(df_before) < timestep:
        return None

    sequence = df_before.tail(timestep)[["hsig", "wind"]].values
    sequence_scaled = scaler.transform(sequence)

    predictions = []
    remaining_steps = steps

    while remaining_steps > 0:
        X_input = sequence_scaled.reshape(1, timestep, 2)

        pred_scaled = model.predict(X_input)

        # reshape ke (4 step, 2 variabel)
        pred_scaled = pred_scaled.reshape(4, 2)

        # balik ke nilai asli
        pred_real = scaler.inverse_transform(pred_scaled)

        for i in range(len(pred_real)):
            hsig = float(pred_real[i, 0])
            wind = float(pred_real[i, 1])

            predictions.append([hsig, wind])

            # update sequence pakai hasil prediksi
            new_scaled = pred_scaled[i]
            sequence_scaled = np.vstack([sequence_scaled[1:], new_scaled])

            remaining_steps -= 1
            if remaining_steps == 0:
                break

    return predictions

# =========================
# LOAD FILE GAMBAR
# =========================
background_base64 = get_base64_image("Background.jpg")
bmkg_base64 = get_base64_image("BMKG.jpeg")
upn_base64 = get_base64_image("UPN.jpeg")

# =========================
# CSS TAMPILAN
# =========================
st.markdown(f"""
<style>
.stApp {{
    background-image: linear-gradient(rgba(5, 10, 20, 0.72), rgba(5, 10, 20, 0.72)),
                      url("data:image/jpg;base64,{background_base64}");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}}

.block-container {{
    padding-top: 4.5rem;
    padding-bottom: 2rem;
}}

.top-wrapper {{
    display: flex;
    flex-direction: row;
    align-items: center;
    justify-content: center;
    gap: 30px;
    margin-top: 30px;
    margin-bottom: 28px;
    flex-wrap: nowrap;
}}

.left-logos {{
    display: flex;
    flex-direction: row;
    align-items: center;
    gap: 22px;
    flex-shrink: 0;
}}

.logo-item {{
    display: flex;
    flex-direction: column;
    align-items: center;
    color: white;
    font-weight: 700;
    font-size: 18px;
    min-width: 95px;
}}

.logo-item img {{
    width: 92px;
    height: 92px;
    object-fit: contain;
    border-radius: 50%;
    background: rgba(255,255,255,0.96);
    padding: 8px;
    box-shadow: 0 4px 16px rgba(0,0,0,0.25);
    margin-bottom: 8px;
}}

.logo-label {{
    line-height: 1.1;
    margin-top: 2px;
}}

.title-box {{
    background: #0b3b8c;
    padding: 20px 34px;
    border-radius: 10px;
    display: block;
    box-shadow: 0 6px 18px rgba(0,0,0,0.28);
    max-width: 820px;
}}

.title-box h1 {{
    color: white;
    margin: 0;
    text-align: center;
    font-size: 32px;
    line-height: 1.3;
    text-transform: uppercase;
}}

.title-box p {{
    color: #eaf2ff;
    margin: 10px 0 0 0;
    text-align: center;
    font-size: 15px;
    line-height: 1.5;
}}

.section-card {{
    background: rgba(8, 18, 38, 0.78);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 18px;
    padding: 20px;
    margin-top: 14px;
    margin-bottom: 20px;
}}

.summary-card {{
    background: rgba(8, 18, 38, 0.78);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 18px;
    padding: 20px;
}}

div[data-testid="stMetric"] {{
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.06);
    padding: 14px;
    border-radius: 14px;
}}

div[data-testid="stForm"] {{
    background: rgba(8, 18, 38, 0.78);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 18px;
    padding: 20px;
}}

div[data-testid="stFormSubmitButton"] > button {{
    background-color: #0b57d0;
    color: white;
    border-radius: 10px;
    border: none;
    padding: 0.6rem 1.5rem;
    font-weight: 700;
}}

div[data-testid="stFormSubmitButton"] > button:hover {{
    background-color: #0a4cb5;
    color: white;
}}

h2, h3 {{
    color: white !important;
}}

p, label, .stMarkdown, .stCaption {{
    color: #f2f5f9;
}}
</style>
""", unsafe_allow_html=True)

# =========================
# LOAD DATA
# =========================
if os.path.exists(DATA_PATH):
    df_raw = pd.read_csv(DATA_PATH)
else:
    df_raw = pd.read_csv("data_nelayan_tuban.csv")
df = preprocess_data(df_raw)

current_row = df.iloc[-1]
current_time = current_row["time"]
current_hsig = float(current_row["hsig"])
current_wind = float(current_row["wind"])
current_status = classify(current_hsig, current_wind)

# =========================
# HEADER ATAS
# =========================
col1, col2 = st.columns([2,6])

with col1:
    logo1, logo2 = st.columns([1, 1], gap="small")
    with logo1:
        st.markdown("<div style='margin-right:20px;'>", unsafe_allow_html=True)
        st.image("BMKG1.png", width=70)
        st.markdown("</div>", unsafe_allow_html=True)
    with logo2:
        st.markdown("<div style='margin-right:20px;'>", unsafe_allow_html=True)
        st.image("UPN1.png", width=70)

with col2:
    st.markdown("""
    <div style="
        background-color:#0b3b8c;
        padding:15px;
        border-radius:10px;
        display:flex;
        justify-content:center;
        align-items:center;
    ">
        <h2 style="color:white; margin:0;">
        Sistem Peringatan Dini Keamanan Berlayar Nelayan
        </h2>
    </div>
    """, unsafe_allow_html=True)

# =========================
# KONDISI SAAT INI
# =========================
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.info("Berbasis parameter ketinggian gelombang dan kecepatan angin")
st.subheader("Kondisi Saat Ini")
show_status_box(current_status, "Status saat ini")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Update Terakhir", current_time.strftime("%Y-%m-%d %H:%M"))
with col2:
    st.metric("Ketinggian Gelombang saat ini", f"{current_hsig:.3f} m")
with col3:
    st.metric("Kecepatan Angin saat ini", f"{current_wind:.3f} knots")
with col4:
    st.metric("Status", current_status)
st.markdown('</div>', unsafe_allow_html=True)

# =========================
# GRAFIK HISTORIS 24 JAM TERAKHIR
# =========================
last_time = df["time"].max()
start_time = last_time - pd.Timedelta(hours=24)

df_24h = df[df["time"] >= start_time].copy()

grafik1, grafik2 = st.columns(2)

with grafik1:
    fig_hsig = px.line(
        df_24h,
        x="time",
        y="hsig",
        title="Ketinggian Gelombang dalam 24 Jam Terakhir",
        markers=True
    )
    fig_hsig.update_layout(
        xaxis_title="Waktu",
        yaxis_title="Ketinggian Gelombang (m)",
        template="plotly_dark",
        height=350
    )
    st.plotly_chart(fig_hsig, use_container_width=True)

with grafik2:
    fig_wind = px.line(
        df_24h,
        x="time",
        y="wind",
        title="Kecepatan Angin dalam 24 Jam Terakhir",
        markers=True
    )
    fig_wind.update_layout(
        xaxis_title="Waktu",
        yaxis_title="Kecepatan Angin (knots)",
        template="plotly_dark",
        height=350
    )
    st.plotly_chart(fig_wind, use_container_width=True)

# =========================
# FORM INPUT
# =========================
st.subheader("Cek Kondisi pada Tanggal dan Jam Tertentu")

with st.form("cek_kondisi_form"):
    col_a, col_b = st.columns(2)

    with col_a:
        selected_date = st.date_input(
            "Pilih tanggal",
            value=current_time.date(),
            min_value=df["time"].min().date(),
            max_value=(df["time"].max() + pd.Timedelta(days=3)).date()
        )

    with col_b:
        available_times = sorted(df["time"].dt.strftime("%H:%M:%S").unique().tolist())
        default_time = current_time.strftime("%H:%M:%S") if current_time.strftime("%H:%M:%S") in available_times else available_times[0]
        selected_time = st.selectbox(
            "Pilih jam",
            options=available_times,
            index=available_times.index(default_time)
        )

    submit_button = st.form_submit_button("Enter")

# =========================
# FUNGSI STEP INDEX (UNTUK PILIH PREDIKSI KE BERAPA)
# =========================
def get_step_index(selected_datetime, df):
    last_time = df["time"].max()
    diff_hours = int((selected_datetime - last_time).total_seconds() // 3600)
    
    if diff_hours < 0:
        return 0
    return min(diff_hours, 23)  # maksimal 24 jam

# =========================
# HASIL CEK
# =========================
if submit_button:
    selected_datetime = pd.to_datetime(f"{selected_date} {selected_time}")

    is_prediction = False  # 🔥 penanda

    # =========================
    # CEK: DATA HISTORIS ATAU PREDIKSI
    # =========================
    if selected_datetime <= df["time"].max():
        # 🔵 AMBIL DATA ASLI
        chosen_row = find_closest_data(df, selected_datetime)

        pred_hsig = float(chosen_row["hsig"])
        pred_wind = float(chosen_row["wind"])

    else:
        # 🔴 PREDIKSI DENGAN MODEL
        preds = forecast_future(df, selected_datetime, steps=24)

        if preds is None:
            st.error("Data historis tidak cukup untuk melakukan prediksi.")
            st.stop()

        diff_hours = int((selected_datetime - df["time"].max()).total_seconds() // 3600)

        if diff_hours < 0:
            diff_hours = 0
        if diff_hours > 23:
            diff_hours = 23

        pred_hsig, pred_wind = preds[diff_hours]
        is_prediction = True  # 🔥 aktifkan grafik

    # =========================
    # TAMPILKAN HASIL
    # =========================
    chosen_status = classify(pred_hsig, pred_wind)

    st.markdown('<div class="summary-card">', unsafe_allow_html=True)
    st.subheader("Hasil Pengecekan Waktu Pilihan")
    show_status_box(chosen_status, "Status pada waktu yang dipilih")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Tanggal & jam dipilih", str(selected_datetime))
    with c2:
        st.metric("Ketinggian Gelombang", f"{pred_hsig:.3f} m")
    with c3:
        st.metric("Kecepatan Angin", f"{pred_wind:.3f} knots")
    with c4:
        st.metric("Status", chosen_status)

    st.markdown("### Ringkasan")
    st.write(
        f"Pada **{selected_datetime}**, **ketinggian gelombang** sebesar "
        f"**{pred_hsig:.3f} m** dan **kecepatan angin** sebesar "
        f"**{pred_wind:.3f} knots**. Berdasarkan ambang batas yang digunakan, "
        f"kondisi ini termasuk **{chosen_status}** untuk berlayar."
    )

    # =========================
    # GRAFIK (BALIK KE BAWAH)
    # =========================
    if is_prediction:
        future_times = pd.date_range(start=selected_datetime, periods=24, freq="h")

        df_future = pd.DataFrame(preds, columns=["hsig", "wind"])
        df_future["time"] = future_times

        fig = px.line(
            df_future,
            x="time",
            y="hsig",
            title="Prediksi Ketinggian Gelombang 24 Jam ke Depan"
        )

        fig.update_layout(template="plotly_dark")

        st.plotly_chart(fig, use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

# =========================
# PREVIEW DATA
# =========================
with st.expander("Lihat preview data"):
    st.dataframe(df.head(20))