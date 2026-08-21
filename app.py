import streamlit as st
from ultralytics import YOLO
import os
from PIL import Image
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av
import cv2

# ── Konfigurasi Halaman ────────────────────────────────────────
st.set_page_config(
    page_title="Deteksi Jenis Sampah",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ─────────────────────────────────────────────────
st.markdown("""
<style>
    /* Import font */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    /* Reset & Base */
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    /* Background utama */
    .stApp {
        background: linear-gradient(135deg, #f0faf4 0%, #ffffff 50%, #e8f5e9 100%);
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1b5e20 0%, #2e7d32 60%, #388e3c 100%) !important;
        border-right: none !important;
    }
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] div,
    [data-testid="stSidebar"] strong,
    [data-testid="stSidebar"] b {
        color: #ffffff !important;
    }
    [data-testid="stSidebar"] hr {
        border-color: rgba(255,255,255,0.2) !important;
    }

    /* Pastikan teks konten utama tetap gelap */
    .main p, .main span, .main div, .main li, .main h1, .main h2, .main h3, .main h4 {
        color: inherit;
    }
    .block-container p {
        color: #212121 !important;
    }
    .block-container li {
        color: #212121 !important;
    }
    .block-container h1, .block-container h2,
    .block-container h3, .block-container h4 {
        color: #1b5e20 !important;
    }

    /* Logo sidebar */
    .sidebar-logo {
        text-align: center;
        padding: 20px 10px 10px 10px;
    }
    .sidebar-logo .logo-icon {
        font-size: 52px;
        display: block;
        margin-bottom: 6px;
    }
    .sidebar-logo .logo-title {
        font-size: 22px;
        font-weight: 800;
        color: #ffffff !important;
        letter-spacing: -0.5px;
    }
    .sidebar-logo .logo-sub {
        font-size: 11px;
        color: #a5d6a7 !important;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-top: 2px;
    }

    /* Tombol sidebar */
    [data-testid="stSidebar"] .stButton > button {
        background: rgba(255,255,255,0.1) !important;
        color: #ffffff !important;
        border: 1px solid rgba(255,255,255,0.2) !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        font-size: 14px !important;
        padding: 10px 16px !important;
        transition: all 0.2s ease !important;
        margin-bottom: 4px !important;
        width: 100% !important;
    }
    [data-testid="stSidebar"] .stButton > button:hover {
        background: rgba(255,255,255,0.25) !important;
        border-color: rgba(255,255,255,0.5) !important;
        transform: translateX(3px) !important;
    }

    /* Slider */
    [data-testid="stSidebar"] .stSlider > div > div > div {
        background: #a5d6a7 !important;
    }

    /* Hero banner */
    .hero-banner {
        background: linear-gradient(135deg, #1b5e20 0%, #2e7d32 40%, #43a047 100%);
        border-radius: 20px;
        padding: 40px 48px;
        margin-bottom: 28px;
        color: white;
        position: relative;
        overflow: hidden;
    }
    .hero-banner::before {
        content: '♻️';
        position: absolute;
        right: 40px;
        top: 50%;
        transform: translateY(-50%);
        font-size: 100px;
        opacity: 0.15;
    }
    .hero-banner h1 {
        font-size: 32px;
        font-weight: 800;
        margin: 0 0 8px 0;
        letter-spacing: -0.5px;
        color: white !important;
    }
    .hero-banner p {
        font-size: 15px;
        color: #c8e6c9 !important;
        margin: 0;
    }

    /* Kartu statistik */
    .stat-card {
        background: white;
        border-radius: 16px;
        padding: 20px 24px;
        border: 1px solid #e8f5e9;
        box-shadow: 0 2px 12px rgba(46,125,50,0.08);
        text-align: center;
        transition: transform 0.2s ease;
    }
    .stat-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(46,125,50,0.15);
    }
    .stat-card .stat-icon {
        font-size: 32px;
        margin-bottom: 8px;
    }
    .stat-card .stat-value {
        font-size: 28px;
        font-weight: 800;
        color: #2e7d32;
        line-height: 1;
    }
    .stat-card .stat-label {
        font-size: 12px;
        color: #78909c;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-top: 4px;
    }

    /* Section header */
    .section-header {
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 16px;
    }
    .section-header .section-icon {
        background: #e8f5e9;
        color: #2e7d32;
        width: 36px;
        height: 36px;
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 18px;
    }
    .section-header h3 {
        font-size: 18px;
        font-weight: 700;
        color: #1b5e20;
        margin: 0;
    }

    /* Card wrapper */
    .content-card {
        background: white;
        border-radius: 16px;
        padding: 24px;
        border: 1px solid #e8f5e9;
        box-shadow: 0 2px 12px rgba(46,125,50,0.06);
        margin-bottom: 16px;
    }

    /* Badge deteksi */
    .detection-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: #e8f5e9;
        border: 1px solid #a5d6a7;
        color: #1b5e20;
        padding: 8px 16px;
        border-radius: 50px;
        font-size: 14px;
        font-weight: 600;
        margin: 4px;
    }

    /* Tabel hasil */
    .result-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 12px 16px;
        border-radius: 10px;
        margin-bottom: 8px;
        background: #f9fbe7;
        border-left: 4px solid #66bb6a;
    }
    .result-row .result-class {
        font-weight: 700;
        color: #1b5e20;
        font-size: 15px;
    }
    .result-row .result-count {
        background: #2e7d32;
        color: white;
        padding: 3px 12px;
        border-radius: 50px;
        font-size: 13px;
        font-weight: 700;
    }

    /* Confidence bar */
    .conf-bar-wrap {
        margin-top: 6px;
    }
    .conf-bar-label {
        font-size: 12px;
        color: #78909c;
        margin-bottom: 3px;
    }
    .conf-bar {
        background: #e8f5e9;
        border-radius: 50px;
        height: 8px;
        overflow: hidden;
    }
    .conf-bar-fill {
        background: linear-gradient(90deg, #43a047, #1b5e20);
        height: 100%;
        border-radius: 50px;
    }

    /* Alert tidak terdeteksi */
    .no-detection {
        text-align: center;
        padding: 32px;
        color: #90a4ae;
    }
    .no-detection .nd-icon {
        font-size: 48px;
        margin-bottom: 12px;
    }
    .no-detection p {
        font-size: 15px;
        margin: 0;
    }

    /* Upload area */
    [data-testid="stFileUploader"] {
        border: 2px dashed #a5d6a7 !important;
        border-radius: 16px !important;
        background: #f1f8e9 !important;
        padding: 8px !important;
    }

    /* Sembunyikan elemen default streamlit (kecuali header agar tombol panah muncul) */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
   
    .block-container {
        padding-top: 24px !important;
        padding-bottom: 24px !important;
        max-width: 1200px !important;
    }

    /* Buat background header menjadi transparan agar menyatu dengan background */
    header[data-testid="stHeader"] {
        background-color: transparent !important;
    }

    /* Tombol panah toggle sidebar jadi HITAM dan dipaksa tampil 100% (tidak transparan) */
    [data-testid="collapsedControl"],
    [data-testid="stSidebarCollapsedControl"],
    [data-testid="stSidebarCollapseButton"] {
        opacity: 1 !important;
        visibility: visible !important;
    }

    /* Targetkan semua elemen gambar panahnya agar full hitam */
    [data-testid="collapsedControl"] svg,
    [data-testid="stSidebarCollapsedControl"] svg,
    [data-testid="stSidebarCollapseButton"] svg,
    [data-testid="collapsedControl"] svg path,
    [data-testid="stSidebarCollapsedControl"] svg path,
    [data-testid="stSidebarCollapseButton"] svg path {
        color: #000000 !important;
        fill: #000000 !important;
        stroke: #000000 !important;
    }
</style>
""", unsafe_allow_html=True)

# ── Load Model ─────────────────────────────────────────────────
@st.cache_resource
def load_model(path):
    try:
        return YOLO(path)
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        return None

MODEL_PATH  = 'best.pt'
model       = load_model(MODEL_PATH)

if not model:
    st.stop()

CLASS_NAMES = model.names

# ── Sidebar ────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("**Menu**")
    st.markdown("---")
    st.markdown("**Pilih Mode**")

    if 'app_mode' not in st.session_state:
        st.session_state.app_mode = "Beranda"

    if st.button("Home", use_container_width=True):
        st.session_state.app_mode = "Beranda"
    if st.button("Deteksi Gambar", use_container_width=True):
        st.session_state.app_mode = "Deteksi Gambar"
    if st.button("Deteksi Real-Time (Webcam)", use_container_width=True):
        st.session_state.app_mode = "Deteksi Real-Time"

    st.markdown("---")
    confidence_threshold = st.slider(
        "Tingkat Keyakinan Deteksi",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
    )


# ==============================================================================
# ── BERANDA ───────────────────────────────────────────────────────────────────
# ==============================================================================
if st.session_state.app_mode == "Beranda":

    # Judul utama
    st.markdown("""
    <div style="display:flex;align-items:center;gap:12px;margin-bottom:8px">
        <span style="font-size:36px">♻️</span>
        <h1 style="font-size:28px;font-weight:800;color:#1b5e20;margin:0">
            Aplikasi Deteksi Sampah Real-Time
        </h1>
    </div>
    <p style="color:#546e7a;font-size:15px;margin-bottom:28px">
        Sistem deteksi dan klasifikasi jenis sampah berbasis YOLO26n.
    </p>
    <hr style="border-color:#e8f5e9;margin-bottom:24px">
    """, unsafe_allow_html=True)

    # Tentang Aplikasi
    st.markdown("#### Tentang Aplikasi Ini")
    st.markdown("""
    Aplikasi ini menggunakan model **YOLO26n** untuk mendeteksi dan mengklasifikasikan
    jenis sampah secara otomatis. Terdapat dua mode penggunaan yang tersedia:
    deteksi melalui gambar yang diunggah, dan deteksi secara real-time menggunakan kamera.
    """)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### Fitur Utama")

    st.markdown("""
    <div class="content-card">
        <ul style="margin:0;padding-left:20px;line-height:2">
            <li><b>Streaming Real-Time (WebRTC):</b> Mendeteksi objek sampah secara langsung melalui kamera dengan latensi rendah.</li>
            <li><b>Deteksi dari Gambar:</b> Pengguna dapat mengunggah gambar untuk dianalisis oleh model.</li>
            <li><b>Klasifikasi Jenis Sampah:</b> Model mampu membedakan sampah <b>Organik</b>, <b>Anorganik</b>, dan <b>B3</b>.</li>
            <li><b>Ringkasan Hasil:</b> Menampilkan jumlah objek terdeteksi, jenis sampah, dan tingkat akurasi deteksi.</li>
            <li><b>Pengaturan Fleksibel:</b> Pengguna dapat menyesuaikan tingkat keyakinan deteksi sesuai kebutuhan.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### Jenis Sampah yang Dideteksi")

    k1, k2, k3 = st.columns(3)
    with k1:
        st.markdown("""
        <div class="content-card" style="border-left:4px solid #4caf50">
            <div style="font-weight:700;color:#1b5e20;font-size:15px;margin-bottom:6px">Sampah Organik</div>
            <div style="font-size:13px;color:#546e7a">Sisa makanan, daun, dan bahan yang dapat terurai secara alami.</div>
        </div>
        """, unsafe_allow_html=True)
    with k2:
        st.markdown("""
        <div class="content-card" style="border-left:4px solid #2196f3">
            <div style="font-weight:700;color:#1b5e20;font-size:15px;margin-bottom:6px">Sampah Anorganik</div>
            <div style="font-size:13px;color:#546e7a">Plastik, kaca, logam, dan bahan yang tidak dapat terurai.</div>
        </div>
        """, unsafe_allow_html=True)
    with k3:
        st.markdown("""
        <div class="content-card" style="border-left:4px solid #f44336">
            <div style="font-weight:700;color:#1b5e20;font-size:15px;margin-bottom:6px">Sampah B3</div>
            <div style="font-size:13px;color:#546e7a">Bahan Berbahaya dan Beracun seperti baterai, obat, dan lampu bekas.</div>
        </div>
        """, unsafe_allow_html=True)


# ==============================================================================
# ── DETEKSI GAMBAR ────────────────────────────────────────────────────────────
# ==============================================================================
elif st.session_state.app_mode == "Deteksi Gambar":

    st.markdown("""
    <div class="hero-banner">
        <h1>🖼️ Deteksi dari Gambar</h1>
        <p>Upload gambar sampah dan biarkan AI menganalisis jenisnya secara otomatis.</p>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Pilih gambar untuk dideteksi",
        type=["jpg", "jpeg", "png"],
        help="Format yang didukung: JPG, JPEG, PNG"
    )

    if uploaded_file is not None:
        image           = Image.open(uploaded_file)
        image_np_rgb    = np.array(image)
        image_np_bgr    = cv2.cvtColor(image_np_rgb, cv2.COLOR_RGB2BGR)

        # Jalankan deteksi
        with st.spinner("🔍 Menganalisis gambar..."):
            results         = model.predict(image_np_bgr, conf=confidence_threshold)
            annotated_bgr   = results[0].plot()
            annotated_rgb   = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

        # Tampilkan gambar
        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            <div class="section-header">
                <div class="section-icon">📷</div>
                <h3>Gambar Asli</h3>
            </div>
            """, unsafe_allow_html=True)
            st.image(image, use_container_width=True)

        with col2:
            st.markdown("""
            <div class="section-header">
                <div class="section-icon">🎯</div>
                <h3>Hasil Deteksi</h3>
            </div>
            """, unsafe_allow_html=True)
            st.image(annotated_rgb, use_container_width=True)

        # ── Hasil Deteksi Detail ─────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        <div class="section-header">
            <div class="section-icon">📊</div>
            <h3>Ringkasan Hasil Deteksi</h3>
        </div>
        """, unsafe_allow_html=True)

        boxes = results[0].boxes

        if len(boxes) > 0:
            # Kumpulkan data per class
            class_data = {}
            for box in boxes:
                class_id   = int(box.cls)
                class_name = CLASS_NAMES[class_id]
                conf_val   = float(box.conf)
                if class_name not in class_data:
                    class_data[class_name] = {"count": 0, "confs": []}
                class_data[class_name]["count"] += 1
                class_data[class_name]["confs"].append(conf_val)

            # Kartu ringkasan angka
            total_obj  = len(boxes)
            total_kelas = len(class_data)
            avg_conf   = sum(float(b.conf) for b in boxes) / total_obj

            m1, m2, m3 = st.columns(3)
            with m1:
                st.markdown(f"""
                <div class="stat-card">
                    <div class="stat-icon">🎯</div>
                    <div class="stat-value">{total_obj}</div>
                    <div class="stat-label">Total Objek Terdeteksi</div>
                </div>
                """, unsafe_allow_html=True)
            with m2:
                st.markdown(f"""
                <div class="stat-card">
                    <div class="stat-icon">🗑️</div>
                    <div class="stat-value">{total_kelas}</div>
                    <div class="stat-label">Jenis Sampah Ditemukan</div>
                </div>
                """, unsafe_allow_html=True)
            with m3:
                st.markdown(f"""
                <div class="stat-card">
                    <div class="stat-icon">✅</div>
                    <div class="stat-value">{avg_conf:.1%}</div>
                    <div class="stat-label">Rata-rata Akurasi</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Detail per class
            st.markdown("**Detail per Jenis Sampah:**")
            for class_name, data in class_data.items():
                count    = data["count"]
                avg_c    = sum(data["confs"]) / len(data["confs"])
                max_c    = max(data["confs"])
                min_c    = min(data["confs"])
                bar_pct  = int(avg_c * 100)

                st.markdown(f"""
                <div class="content-card" style="margin-bottom:12px">
                    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px">
                        <div style="font-weight:700;color:#1b5e20;font-size:16px">🗑️ {class_name}</div>
                        <span class="result-count" style="background:#2e7d32;color:white;padding:4px 14px;border-radius:50px;font-size:13px;font-weight:700">{count} objek</span>
                    </div>
                    <div style="display:flex;gap:24px;margin-bottom:12px">
                        <div style="text-align:center">
                            <div style="font-size:20px;font-weight:800;color:#2e7d32">{avg_c:.1%}</div>
                            <div style="font-size:11px;color:#90a4ae;text-transform:uppercase;letter-spacing:0.5px">Rata-rata Conf.</div>
                        </div>
                        <div style="text-align:center">
                            <div style="font-size:20px;font-weight:800;color:#43a047">{max_c:.1%}</div>
                            <div style="font-size:11px;color:#90a4ae;text-transform:uppercase;letter-spacing:0.5px">Tertinggi</div>
                        </div>
                        <div style="text-align:center">
                            <div style="font-size:20px;font-weight:800;color:#ef6c00">{min_c:.1%}</div>
                            <div style="font-size:11px;color:#90a4ae;text-transform:uppercase;letter-spacing:0.5px">Terendah</div>
                        </div>
                    </div>
                    <div class="conf-bar-label">Tingkat Kepercayaan Rata-rata</div>
                    <div class="conf-bar">
                        <div class="conf-bar-fill" style="width:{bar_pct}%"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        else:
            st.markdown("""
            <div class="content-card">
                <div class="no-detection">
                    <div class="nd-icon">🔍</div>
                    <p><b>Tidak ada objek sampah yang terdeteksi.</b></p>
                    <p style="margin-top:8px;font-size:13px">Coba turunkan nilai confidence threshold atau gunakan gambar yang lebih jelas.</p>
                </div>
            </div>
            """, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div class="content-card" style="text-align:center;padding:48px">
            <div style="font-size:64px;margin-bottom:16px">📤</div>
            <div style="font-weight:700;color:#1b5e20;font-size:18px;margin-bottom:8px">Upload Gambar untuk Memulai</div>
            <div style="font-size:14px;color:#78909c">Pilih file gambar di atas untuk menjalankan deteksi sampah</div>
        </div>
        """, unsafe_allow_html=True)


# ==============================================================================
# ── DETEKSI REAL-TIME ─────────────────────────────────────────────────────────
# ==============================================================================
elif st.session_state.app_mode == "Deteksi Real-Time":

    st.markdown("""
    <div class="hero-banner">
        <h1>📷 Deteksi Real-Time</h1>
        <p>Aktifkan kamera dan arahkan ke sampah — AI akan mendeteksi secara langsung.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="content-card" style="margin-bottom:20px">
        <div style="display:flex;gap:12px;align-items:flex-start">
            <div style="font-size:20px">💡</div>
            <div>
                <div style="font-weight:700;color:#1b5e20;margin-bottom:4px">Petunjuk Penggunaan</div>
                <div style="font-size:13px;color:#546e7a">
                    Klik tombol <b>START</b> untuk mengaktifkan kamera. Pastikan browser mengizinkan akses kamera.
                    Arahkan kamera ke objek sampah dan tunggu hasil deteksi muncul secara otomatis.
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    class YOLOVideoProcessor(VideoProcessorBase):
        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            image           = frame.to_ndarray(format="bgr24")
            results         = model.predict(image, conf=confidence_threshold)
            annotated_frame = results[0].plot()
            return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        webrtc_streamer(
            key="yolo-webrtc",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=YOLOVideoProcessor,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
            rtc_configuration={
                "iceServers": [
                    {"urls": ["stun:stun.l.google.com:19302"]},
                    {
                        "urls": ["turn:openrelay.metered.ca:80"],
                        "username": "openrelayproject",
                        "credential": "openrelayproject",
                    },
                ]
            }
        )

    st.markdown("""
    <div style="text-align:center;margin-top:16px;font-size:13px;color:#90a4ae">
        Menggunakan WebRTC untuk streaming real-time dengan latensi rendah
    </div>
    """, unsafe_allow_html=True)
