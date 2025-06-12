import streamlit as st
import cv2
import numpy as np
from PIL import Image
from collections import Counter # BARU: Untuk menghitung penyakit yang paling umum

from ultralytics import YOLO

# Page config
st.set_page_config(
    page_title="🌺 Orchid Disease Detection",
    page_icon="🌺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inisialisasi session_state untuk kontrol kamera
if 'camera_activated' not in st.session_state:
    st.session_state.camera_activated = False

# ... (CSS tetap sama, tidak perlu diubah) ...
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4, #45B7D1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    
    .feature-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
        transition: transform 0.3s ease;
    }
    
    .feature-card:hover {
        transform: translateY(-5px);
    }
    
    .detection-result {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 2rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        margin: 1.5rem 0;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    
    .recommendation-card {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin-top: 1rem;
        color: white;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    }
    
    .healthy-result {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 2rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        margin: 1.5rem 0;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    
    .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-weight: bold;
        transition: all 0.3s ease;
        width: 100%; /* Make button full width */
    }
    
    .stButton > button:hover {
        transform: scale(1.05);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
</style>
""", unsafe_allow_html=True)

# Disease information database (tetap sama)
DISEASE_INFO = {
    "Petal Blight": {
        "description": "Penyakit layu kelopak yang disebabkan oleh jamur Botrytis cinerea",
        "symptoms": ["Bercak coklat pada kelopak bunga", "Kelopak menjadi lembek dan berlendir", "Bunga gugur sebelum waktunya"],
        "prevention": [
            "Jaga sirkulasi udara yang baik",
            "Hindari kelembaban berlebih di sekitar bunga",
            "Buang bunga yang sudah layu segera",
            "Semprot fungisida preventif saat musim hujan"
        ],
        "treatment": [
            "Potong semua bagian bunga yang terinfeksi",
            "Aplikasikan fungisida berbahan aktif iprodione",
            "Kurangi penyiraman dan kelembaban",
            "Perbaiki ventilasi udara di sekitar tanaman",
            "Semprot dengan larutan baking soda 1%"
        ]
    },
    "Brown Spot": {
        "description": "Busuk coklat yang disebabkan oleh jamur Monilinia fructicola",
        "symptoms": ["Bercak coklat pada pseudobulb", "Jaringan menjadi lunak dan busuk", "Muncul spora berwarna coklat"],
        "prevention": [
            "Jaga kebersihan area sekitar tanaman",
            "Hindari luka pada pseudobulb",
            "Pastikan drainase media tanam baik",
            "Sterilisasi alat potong sebelum digunakan"
        ],
        "treatment": [
            "Potong bagian yang terinfeksi hingga jaringan sehat",
            "Aplikasikan pasta fungisida pada luka",
            "Ganti media tanam yang terkontaminasi",
            "Semprot dengan fungisida tembaga",
            "Isolasi tanaman untuk mencegah penyebaran"
        ]
    },
    "Soft Rot": {
        "description": "Busuk lunak yang disebabkan oleh bakteri Erwinia carotovora",
        "symptoms": ["Jaringan tanaman menjadi lembek", "Bau busuk yang menyengat", "Bagian yang terinfeksi berubah warna"],
        "prevention": [
            "Hindari penyiraman berlebihan",
            "Jaga kebersihan alat dan media tanam",
            "Pastikan sirkulasi udara baik",
            "Hindari melukai tanaman saat perawatan"
        ],
        "treatment": [
            "Potong semua bagian yang terinfeksi",
            "Keringkan luka dengan kertas tissue",
            "Aplikasikan bakterisida streptomycin",
            "Kurangi kelembaban sekitar tanaman",
            "Ganti media tanam dengan yang steril"
        ]
    }
}


@st.cache_resource
def load_model():
    """Load YOLO model from .pt file"""
    try:
        model = YOLO("best.pt")
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None

# DIUBAH: Fungsi ini sekarang mengembalikan LIST dari semua deteksi
def predict_disease_yolo(model, image):
    """Make prediction using YOLO model and return all detections."""
    if model is None: return []
    try:
        if isinstance(image, Image.Image):
            image = np.array(image.convert("RGB"))
        
        results = model(image, conf=0.25)
        detections = []
        
        for result in results:
            if result.boxes is not None and len(result.boxes) > 0:
                confidences = result.boxes.conf.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy()
                boxes = result.boxes.xyxy.cpu().numpy()
                
                # BARU: Loop melalui semua box yang terdeteksi
                for i in range(len(boxes)):
                    detection = {
                        "disease": model.names[int(classes[i])],
                        "confidence": float(confidences[i]),
                        "box": boxes[i]
                    }
                    detections.append(detection)
        
        return detections # Mengembalikan list, bisa kosong jika tidak ada deteksi
        
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return []

# DIUBAH: Fungsi ini sekarang menerima LIST deteksi untuk digambar
def draw_detection_on_image(image, detections):
    """Draw all bounding boxes and labels on the image."""
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)

    cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # BARU: Loop melalui setiap deteksi dalam list
    for detection in detections:
        box = detection["box"]
        disease = detection["disease"]
        confidence = detection["confidence"]
        
        x1, y1, x2, y2 = map(int, box)
        color = (0, 0, 255) # Red for disease
        cv2.rectangle(cv_image, (x1, y1), (x2, y2), color, 2)
        
        label = f"{disease}: {confidence:.1%}"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(cv_image, (x1, y1 - label_size[1] - 10), (x1 + label_size[0], y1), color, -1)
        cv2.putText(cv_image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
    return cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

def display_disease_info(disease_name):
    """Display disease information and recommendations."""
    if disease_name in DISEASE_INFO:
        info = DISEASE_INFO[disease_name]
        st.markdown(f"### 📋 Informasi & Rekomendasi untuk: {disease_name}")
        st.write(f"**Deskripsi:** {info['description']}")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("#### 🔍 Gejala Umum")
            for symptom in info['symptoms']: st.write(f"• {symptom}")
        with col2:
            st.markdown("#### 🛡️ Cara Pencegahan")
            for prevention in info['prevention']: st.write(f"• {prevention}")
        with col3:
            st.markdown("#### 💊 Cara Pengobatan")
            for treatment in info['treatment']: st.write(f"• {treatment}")

def main():
    st.markdown('<h1 class="main-header">🌺 Orchid Disease Detection System</h1>', unsafe_allow_html=True)
    with st.sidebar:
        # ... (Sidebar tetap sama) ...
        st.markdown("""<div class="sidebar-content"><h2>🎯 Fitur Aplikasi</h2><p>Sistem AI untuk mendeteksi penyakit tanaman anggrek menggunakan model YOLO.</p></div>""", unsafe_allow_html=True)
        st.markdown("### 📋 Penyakit yang Dapat Dideteksi:")
        st.write("🦠 Petal Blight"); st.write("🍃 Brown Spot"); st.write("🌿 Soft Rot")
        st.markdown("### 📊 Akurasi Model:"); st.progress(0.89); st.write("Akurasi Rata-rata 89%")
        st.markdown("### 💡 Tips Penggunaan:"); st.info("Gunakan foto dengan pencahayaan baik, fokus pada area yang terinfeksi, dan pastikan gambar tidak buram.")

    model = load_model()
    tab1, tab2 = st.tabs(["📷 Camera Capture", "📤 Upload Gambar"])
    
    # DIUBAH: Logika di dalam tab disesuaikan untuk menangani list deteksi
    def process_and_display_results(image):
        with st.spinner("Menganalisis gambar..."):
            detections = predict_disease_yolo(model, image)
            
            st.markdown("---"); st.subheader("Hasil Analisis")
            
            col_res1, col_res2 = st.columns(2)
            with col_res1:
                st.image(image, caption="Gambar Asli", use_container_width=True)
            with col_res2:
                annotated_image = draw_detection_on_image(image, detections)
                st.image(annotated_image, caption="Hasil Deteksi AI", use_container_width=True)
            
            # BARU: Logika untuk menangani hasil (sehat vs sakit)
            if not detections:
                st.markdown(f"""<div class="healthy-result"><h2>✅ Tanaman Sehat!</h2><p>Tidak ditemukan spot penyakit pada gambar.</p></div>""", unsafe_allow_html=True)
            else:
                disease_names = [d['disease'] for d in detections]
                most_common_disease = Counter(disease_names).most_common(1)[0][0]
                
                st.markdown(f"""
                <div class="detection-result">
                    <h2>⚠️ Penyakit Terdeteksi!</h2>
                    <p>Total ditemukan <strong>{len(detections)}</strong> spot penyakit.</p>
                    <p>Jenis penyakit terdeteksi: <strong>{', '.join(set(disease_names))}</strong></p>
                </div>
                """, unsafe_allow_html=True)
                
                # Menampilkan rekomendasi untuk penyakit yang paling umum
                display_disease_info(most_common_disease)

    with tab1:
        st.markdown("""<div class="feature-card"><h3>📷 Deteksi dengan Camera Capture</h3><p>Ambil foto menggunakan kamera untuk analisis penyakit secara langsung.</p></div>""", unsafe_allow_html=True)
        
        if not st.session_state.camera_activated:
            if st.button("📷 Aktifkan Kamera"):
                st.session_state.camera_activated = True
                st.rerun()
        else:
            st.info("Kamera aktif. Silakan posisikan tanaman dan ambil foto.")
            camera_input = st.camera_input("Arahkan kamera...", key="camera", label_visibility="collapsed")
            
            if st.button("❌ Matikan Kamera"):
                st.session_state.camera_activated = False
                st.rerun()

            if camera_input is not None:
                image = Image.open(camera_input)
                process_and_display_results(image)
    
    with tab2:
        st.markdown("""<div class="feature-card"><h3>📤 Upload Gambar</h3><p>Upload foto tanaman anggrek dari galeri Anda untuk dianalisis.</p></div>""", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Pilih gambar tanaman anggrek", type=['jpg', 'jpeg', 'png'])
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            # Menampilkan gambar sebelum analisis di layout yang lebih baik
            # st.image(image, caption="Pratinjau Gambar", width=400)
            
            if st.button("🔍 Analisis Penyakit", key="upload_analyze"):
                process_and_display_results(image)
    
    st.markdown("---"); st.markdown("""<div style="text-align: center; color: #666; padding: 2rem;"><p>🌺 Orchid Disease Detection System | Powered by AI & YOLO</p><p>Developed with ❤️ for orchid enthusiasts</p></div>""", unsafe_allow_html=True)

if __name__ == "__main__":
    main()