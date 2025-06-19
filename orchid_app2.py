import streamlit as st
import cv2
import numpy as np
from PIL import Image
from collections import Counter
from ultralytics import YOLO

# Konfigurasi Halaman
st.set_page_config(
    page_title="🌺 Deteksi Penyakit Anggrek",
    page_icon="🌺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inisialisasi session_state untuk kontrol kamera
if 'camera_activated' not in st.session_state:
    st.session_state.camera_activated = False

# CSS untuk desain modern dan responsif (TIDAK DIUBAH)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    :root {
        --primary-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        --success-gradient: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        --warning-gradient: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        --info-gradient: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        --dark-bg: #1a1a1a;
        --card-bg: #2d2d2d;
        --text-primary: #ffffff;
        --text-secondary: #b3b3b3;
        --border-color: #404040;
        --shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
        --shadow-hover: 0 15px 40px rgba(0, 0, 0, 0.4);
    }
    
    .stApp {
        background: linear-gradient(135deg, #0c0c0c 0%, #1a1a1a 50%, #2d2d2d 100%);
        font-family: 'Inter', sans-serif;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    .main-header {
        font-size: clamp(2rem, 5vw, 3.5rem);
        font-weight: 700;
        text-align: center;
        background: linear-gradient(45deg, #ff6b6b, #4ecdc4, #45b7d1, #96ceb4, #feca57);
        background-size: 300% 300%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 2rem 0;
        animation: gradientShift 6s ease-in-out infinite;
        letter-spacing: -0.02em;
    }
    
    @keyframes gradientShift {
        0%, 100% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
    }
    
    .feature-card {
        background: var(--card-bg);
        border: 1px solid var(--border-color);
        padding: 2rem;
        border-radius: 20px;
        color: var(--text-primary);
        margin: 1.5rem 0;
        box-shadow: var(--shadow);
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    
    .feature-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.1), transparent);
        transition: left 0.5s;
    }
    
    .feature-card:hover {
        transform: translateY(-8px);
        box-shadow: var(--shadow-hover);
        border-color: #667eea;
    }
    
    .feature-card:hover::before {
        left: 100%;
    }
    
    .detection-result {
        background: var(--card-bg);
        border: 2px solid #f5576c;
        padding: 2.5rem;
        border-radius: 25px;
        color: var(--text-primary);
        text-align: center;
        margin: 2rem 0;
        box-shadow: 0 0 30px rgba(245, 87, 108, 0.3);
        position: relative;
        overflow: hidden;
    }
    
    .healthy-result {
        background: var(--card-bg);
        border: 2px solid #38ef7d;
        padding: 2.5rem;
        border-radius: 25px;
        color: var(--text-primary);
        text-align: center;
        margin: 2rem 0;
        box-shadow: 0 0 30px rgba(56, 239, 125, 0.3);
        position: relative;
        overflow: hidden;
    }
    
    .recommendation-card {
        background: var(--card-bg);
        border: 1px solid #4facfe;
        padding: 2rem;
        border-radius: 20px;
        margin-top: 2rem;
        color: var(--text-primary);
        box-shadow: 0 0 20px rgba(79, 172, 254, 0.2);
    }
    .recommendation-card h4 {
        color: #5dade2;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .recommendation-card ul {
        list-style: none;
        padding: 0;
    }
    .recommendation-card li {
        background: rgba(102, 126, 234, 0.1);
        padding: 1rem;
        border-radius: 12px;
        margin-bottom: 0.75rem;
        border-left: 4px solid #5dade2;
        transition: all 0.2s ease;
    }
    .recommendation-card li:hover {
        background: rgba(102, 126, 234, 0.2);
        transform: translateX(5px);
    }
    
    .sidebar-content {
        background: var(--card-bg);
        border: 1px solid var(--border-color);
        padding: 1.5rem;
        border-radius: 15px;
        color: var(--text-primary);
        margin: 1rem 0;
        box-shadow: var(--shadow);
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        width: 100%;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        position: relative;
        overflow: hidden;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.6);
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
    
    .disease-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
        gap: 1.5rem;
        margin: 2rem 0;
    }
    
    .disease-card {
        background: var(--card-bg);
        border: 1px solid var(--border-color);
        border-radius: 20px;
        padding: 2rem;
        transition: all 0.3s ease;
    }
    
    .disease-card:hover {
        transform: translateY(-5px);
        box-shadow: var(--shadow-hover);
        border-color: #667eea;
    }
    
    .disease-card h4 {
        color: #5dade2;
        font-size: 1.3rem;
        margin-bottom: 1rem;
        font-weight: 600;
    }
    
    .disease-icon {
        font-size: 3rem;
        margin-bottom: 1rem;
        display: block;
    }
    
    .info-card-grid {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 1.5rem;
        margin-top: 2rem;
    }
    
    .info-card, .info-card-treatment {
        background: var(--card-bg);
        color: var(--text-primary);
        padding: 2rem;
        border-radius: 20px;
        border: 1px solid var(--border-color);
        box-shadow: var(--shadow);
    }
    
    .info-card-treatment {
        grid-column: 1 / -1;
    }
    
    .info-card h4, .info-card-treatment h4 {
        font-size: 1.3rem;
        font-weight: 600;
        text-align: center;
        margin-bottom: 1.5rem;
        color: #5dade2;
        border-bottom: 2px solid var(--border-color);
        padding-bottom: 1rem;
    }
    
    .info-card ul, .info-card-treatment ul {
        list-style: none;
        padding: 0;
    }
    
    .info-card li, .info-card-treatment li {
        background: rgba(102, 126, 234, 0.1);
        padding: 1rem;
        border-radius: 12px;
        margin-bottom: 0.75rem;
        border-left: 4px solid #5dade2;
    }
    
    .dos-donts-list {
        list-style: none;
        padding: 0;
    }
    
    .dos-donts-list li {
        margin-bottom: 0.75rem;
        padding: 1rem;
        border-radius: 12px;
    }
    
    .dos {
        background: rgba(46, 204, 113, 0.1);
        border-left: 4px solid #2ecc71;
    }
    
    .donts {
        background: rgba(231, 76, 60, 0.1);
        border-left: 4px solid #e74c3c;
    }

    [data-testid="stCameraInput"] {
        max-width: 720px;
        margin: 20px auto;
        border: 2px dashed var(--border-color);
        border-radius: 20px;
        padding: 1rem;
    }

    [data-testid="stCameraInput"] video {
        width: 100%;
        height: auto;
        border-radius: 15px;
    }
    
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea, #764ba2);
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: var(--card-bg);
        border-radius: 12px;
        border: 1px solid var(--border-color);
        color: var(--text-primary);
        padding: 0.5rem 1rem;
        font-weight: 500;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    @media (max-width: 768px) {
        .info-card-grid, .disease-grid, .tips-grid {
            grid-template-columns: 1fr;
            gap: 1rem;
        }
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# KONFIGURASI LOGIKA BAHASA DAN DATABASE PENYAKIT
# ==============================================================================

DISEASE_INFO = {
    "Busuk Bunga": {
        "description": "Busuk bunga (petal blight) disebabkan oleh jamur seperti Botrytis cinerea (jamur abu-abu) atau Phytophthora. Penyakit ini menyerang kuncup dan bunga, menyebabkan kerugian signifikan.",
        "symptoms": ["Bercak basah berwarna coklat muda pada kelopak bunga.","Pada infeksi Botrytis, bisa muncul spora abu-abu yang seperti debu.","Kuncup bunga bisa membusuk dan gagal mekar."],
        "prevention": ["Buang bunga yang sudah layu atau terinfeksi secepatnya.","Tingkatkan sirkulasi udara di sekitar bunga.","Hindari menyemprotkan air langsung ke bunga."],
        "treatment": ["Gunakan fungisida yang efektif untuk Botrytis atau Phytophthora.","Potong dan musnahkan semua bagian yang terinfeksi.","Lakukan penyemprotan preventif jika kondisi sangat lembab."]
    },
    "Bercak Coklat": {
        "description": "Busuk Coklat (Brown Spot / Brown Rot) adalah penyakit merusak yang disebabkan oleh jamur (seperti Phytophthora) atau bakteri (seperti Erwinia), menyerang daun dan pseudobulb.",
        "symptoms": ["Bercak basah pada daun yang awalnya kuning-coklat.","Bercak dengan cepat membesar dan berubah menjadi coklat tua atau hitam.","Dapat menyebar ke akar dan menyebabkan busuk akar."],
        "prevention": ["Jaga sirkulasi udara yang baik.","Hindari daun basah terlalu lama, jangan menyiram dari atas.","Gunakan alat potong yang steril."],
        "treatment": ["Segera potong bagian yang terinfeksi hingga ke jaringan sehat dengan alat steril.","Oleskan fungisida/bakterisida pada luka potongan.","Isolasi tanaman yang sakit."]
    },
    "Busuk Lunak": {
        "description": "Busuk Lunak (Soft Rot) adalah penyakit bakteri yang sangat berbahaya dan cepat menyebar, disebabkan oleh Pectobacterium atau Dickeya. Penyakit ini seringkali fatal.",
        "symptoms": ["Daun menjadi bening, basah, dan lembek seperti agar-agar.","Mengeluarkan bau busuk yang sangat khas.","Penyebaran sangat cepat, bisa menghancurkan seluruh tanaman dalam hitungan hari."],
        "prevention": ["Jaga agar daun selalu kering, siram hanya media tanam.","Tingkatkan sirkulasi udara secara maksimal.","Hindari luka mekanis pada tanaman."],
        "treatment": ["Ini darurat! Segera potong seluruh bagian yang terinfeksi.","Gunakan pisau yang disterilkan dengan api/alkohol untuk setiap potongan.","Oleskan bubuk bakterisida/fungisida berbasis tembaga pada luka."]
    }
}

# Daftar kelas dalam BAHASA INGGRIS untuk mencocokkan output model
DISEASE_CLASSES = ["Petal Blight", "Brown Spot", "Soft Rot"]

# Kamus untuk MENERJEMAHKAN hasil model ke Bahasa Indonesia
TRANSLATION_MAP = {
    "Petal Blight": "Busuk Bunga",
    "Brown Spot": "Bercak Coklat",
    "Soft Rot": "Busuk Lunak"
}

@st.cache_resource
def load_model():
    """Memuat model YOLO dari file .pt"""
    try:
        model = YOLO("best.pt")
        return model
    except Exception as e:
        st.error(f"❌ Gagal memuat model: {str(e)}")
        return None

def predict_disease_yolo(model, image):
    """Membuat prediksi dan menerjemahkan hasilnya."""
    if model is None:
        return []
    try:
        if isinstance(image, Image.Image):
            image = np.array(image.convert("RGB"))
        results = model(image, conf=0.25)
        detections = []
        for result in results:
            if result.boxes is not None and len(result.boxes) > 0:
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    class_name_en = model.names[class_id]
                    if class_name_en in DISEASE_CLASSES:
                        class_name_id = TRANSLATION_MAP.get(class_name_en, class_name_en)
                        detection = {
                            "disease": class_name_id,
                            "confidence": float(box.conf[0]),
                            "box": box.xyxy[0].cpu().numpy(),
                        }
                        detections.append(detection)
        return detections
    except Exception as e:
        st.error(f"Gagal melakukan prediksi: {str(e)}")
        return []

def draw_detection_on_image(image, detections):
    """Menggambar kotak deteksi pada gambar."""
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    for detection in detections:
        box = detection["box"]
        disease = detection["disease"]
        confidence = detection["confidence"]
        x1, y1, x2, y2 = map(int, box)
        color = (0, 0, 255)
        cv2.rectangle(cv_image, (x1, y1), (x2, y2), color, 3)
        label = f"{disease}: {confidence:.1%}"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(cv_image, (x1, y1 - label_size[1] - 15), (x1 + label_size[0] + 10, y1), color, -1)
        cv2.putText(cv_image, label, (x1 + 5, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

def analyze_detections(detections):
    """Menganalisis deteksi untuk menentukan status."""
    if not detections:
        return "no_disease_detected", "Tidak ada penyakit yang terdeteksi pada tanaman.", []
    else:
        disease_names = [d['disease'] for d in detections]
        return "diseased", f"Terdeteksi {len(disease_names)} area penyakit.", disease_names

def display_disease_info(disease_name):
    """Menampilkan informasi dan rekomendasi penyakit."""
    info = DISEASE_INFO.get(disease_name)
    if not info:
        return
    st.markdown(f"### 📋 Informasi & Rekomendasi untuk: **{disease_name}**")
    symptoms_list = ''.join([f"<li>{symptom}</li>" for symptom in info['symptoms']])
    prevention_list = ''.join([f"<li>{prevention}</li>" for prevention in info['prevention']])
    treatment_list = ''.join([f"<li>{treatment}</li>" for treatment in info['treatment']])
    card_html = f"""
    <div class="info-card-grid">
        <div class="info-card"><h4>🔍 Gejala Umum</h4><ul>{symptoms_list}</ul></div>
        <div class="info-card"><h4>🛡️ Metode Pencegahan</h4><ul>{prevention_list}</ul></div>
        <div class="info-card-treatment"><h4>💊 Metode Penanganan</h4><ul>{treatment_list}</ul></div>
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)

def process_and_display_results(image, model):
    """Memproses gambar, menjalankan prediksi, dan menampilkan hasil."""
    with st.spinner("🔍 Menganalisis gambar anggrek Anda..."):
        detections = predict_disease_yolo(model, image)
    st.markdown("---")
    st.subheader("📊 Hasil Analisis")
    col_res1, col_res2 = st.columns(2)
    with col_res1:
        st.image(image, caption="📷 Gambar Asli", use_container_width=True)
    with col_res2:
        annotated_image = draw_detection_on_image(image, detections)
        st.image(annotated_image, caption="🤖 Hasil Deteksi AI", use_container_width=True)
    status, message, diseases_found = analyze_detections(detections)
    if status == "no_disease_detected":
        st.markdown(f"""<div class="healthy-result"><h2>✅ Kabar Baik!</h2><h3>Tidak Ada Penyakit Terdeteksi</h3><p style="font-size: 1.1rem; margin: 1rem 0;">{message}</p><p>Anggrek Anda tampak sehat. Pertahankan perawatan yang baik! 🌟</p></div>""", unsafe_allow_html=True)
        st.markdown("""
        <div class="recommendation-card">
            <h4>🌱 Tips Perawatan Anggrek Sehat</h4>
            <ul>
                <li><strong>💡 Pencahayaan:</strong> Sinar matahari tidak langsung yang terang.</li>
                <li><strong>💧 Penyiraman:</strong> Siram saat media tanam hampir kering.</li>
                <li><strong>🌡️ Kelembaban:</strong> Anggrek menyukai kelembaban 50-70%.</li>
                <li><strong>🌬️ Sirkulasi Udara:</strong> Sirkulasi udara yang baik penting untuk mencegah jamur.</li>
                <li><strong>🌿 Pemupukan:</strong> Gunakan pupuk seimbang seminggu sekali saat musim tanam.</li>
                <li><strong>🧐 Inspeksi Rutin:</strong> Periksa tanaman secara teratur untuk tanda-tanda awal penyakit.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    elif status == "diseased":
        most_common_disease = Counter(diseases_found).most_common(1)[0][0]
        st.markdown(f"""<div class="detection-result"><h2>⚠️ Penyakit Terdeteksi!</h2><p>{message}</p><p>Jenis penyakit: <strong>{most_common_disease}</strong></p><p>Segera ambil tindakan untuk mencegah penyebaran!</p></div>""", unsafe_allow_html=True)
        display_disease_info(most_common_disease)

def main():
    st.markdown('<h1 class="main-header">🌺 Sistem Deteksi Penyakit Anggrek</h1>', unsafe_allow_html=True)
    with st.sidebar:
        st.markdown("""<div class="sidebar-content"><h2>🎯 Fitur Aplikasi</h2><p>Sistem berbasis AI untuk mendeteksi penyakit spesifik pada tanaman anggrek.</p></div>""", unsafe_allow_html=True)
        st.markdown("### 🦠 Penyakit yang Dapat Dideteksi:")
        diseases = [
            ("Busuk Bunga", "Busuk pada bunga akibat jamur."),
            ("Bercak Coklat", "Busuk coklat pada daun & pseudobulb."),
            ("Busuk Lunak", "Busuk lunak bakteri yang fatal.")
        ]
        for disease, desc in diseases:
            st.markdown(f"**🌸 {disease}**")
            st.caption(desc)
        st.markdown("### 📊 Performa Model:")
        st.progress(0.89)
        st.markdown("**Akurasi Rata-rata: 89%**")
        st.markdown("### 💡 Tips Pro:")
        st.info("🔍 Gunakan pencahayaan yang baik\n\n📸 Fokus pada area yang terinfeksi\n\n🎯 Hindari gambar buram")
    model = load_model()
    tab_beranda, tab_camera, tab_upload = st.tabs(["🏠 Beranda", "📷 Kamera", "📤 Unggah"])
    with tab_beranda:
        st.markdown("""
        <div style="text-align: center; margin: 2rem 0;">
            <h2 style="color: #5dade2;">🤖 Asisten Perawatan Anggrek AI</h2>
            <p>Selamat datang! Sistem AI kami membantu Anda mengidentifikasi penyakit lebih awal untuk menjaga anggrek Anda tetap sehat.</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("---")
        st.markdown("## 🎯 Penyakit yang Dapat Dideteksi")
        st.markdown("""
        <div class="disease-grid">
            <div class="disease-card"><div class="disease-icon">🌸</div><h4>Busuk Bunga</h4><p>Menyerang kuncup dan bunga, menyebabkan bercak basah dan busuk.</p></div>
            <div class="disease-card"><div class="disease-icon">🍃</div><h4>Bercak Coklat</h4><p>Menyebabkan bercak coklat kehitaman yang basah pada daun dan batang.</p></div>
            <div class="disease-card"><div class="disease-icon">🌿</div><h4>Busuk Lunak</h4><p>Infeksi bakteri yang cepat dan membuat jaringan tanaman lunak serta berbau.</p></div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("---")
        st.markdown("## 📸 Praktik Terbaik untuk Deteksi Akurat")
        st.markdown("""
        <div class="tips-grid">
            <div class="tips-card">
                <h4 style="color: #2ecc71; text-align: center; margin-bottom: 1.5rem;">✅ Lakukan Ini</h4>
                <ul class="dos-donts-list">
                    <li class="dos">🌟 Gunakan pencahayaan alami yang terang</li>
                    <li class="dos">🎯 Fokus pada satu area spesifik</li>
                    <li class="dos">📱 Jaga kestabilan perangkat agar tidak buram</li>
                    <li class="dos">🖼️ Gunakan latar belakang polos jika bisa</li>
                    <li class="dos">📏 Ambil gambar dari jarak yang cukup dekat</li>
                </ul>
            </div>
            <div class="tips-card">
                <h4 style="color: #e74c3c; text-align: center; margin-bottom: 1.5rem;">❌ Hindari Ini</h4>
                <ul class="dos-donts-list">
                    <li class="donts">🌑 Mengambil foto di tempat gelap atau terlalu terang</li>
                    <li class="donts">🏞️ Terlalu banyak bagian tanaman dalam satu gambar</li>
                    <li class="donts">💫 Menggunakan gambar yang buram atau pecah</li>
                    <li class="donts">🌫️ Bayangan menutupi area yang terinfeksi</li>
                    <li class="donts">📐 Sudut ekstrem yang mengubah bentuk tanaman</li>
                </ul>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with tab_camera:
        st.markdown("<h3>📷 Deteksi dengan Kamera</h3>", unsafe_allow_html=True)
        if st.session_state.get('camera_activated', False):
            camera_input = st.camera_input("Arahkan kamera ke tanaman anggrek...", key="camera", label_visibility="collapsed")
            if camera_input is not None:
                image = Image.open(camera_input)
                process_and_display_results(image, model)
            if st.button("❌ Nonaktifkan Kamera"):
                st.session_state.camera_activated = False
                st.rerun()
        else:
            if st.button("📷 Aktifkan Kamera"):
                st.session_state.camera_activated = True
                st.rerun()
    with tab_upload:
        st.markdown("<h3>📤 Unggah Gambar</h3>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Pilih gambar anggrek", type=['jpg', 'jpeg', 'png'])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            if st.button("🔍 Analisis Penyakit", key="upload_analyze"):
                process_and_display_results(image, model)

if __name__ == "__main__":
    main()