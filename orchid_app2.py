import streamlit as st
import cv2
import numpy as np
from PIL import Image
from collections import Counter

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

# CSS styling (tetap sama)
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
    
    .no-plant-result {
        background: linear-gradient(135deg, #ffeaa7 0%, #fab1a0 100%);
        padding: 2rem;
        border-radius: 20px;
        color: #2d3436;
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
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: scale(1.05);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
</style>
""", unsafe_allow_html=True)

# Disease information database (tetap sama untuk penyakit)
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

# FIXED: Definisi kelas sehat dan penyakit dengan nama yang sesuai model
HEALTHY_CLASSES = ["healthy_leaf", "healthy_flower", "Healty Leaf", "Healty Flower", "Healthy Leaf", "Healthy Flower"]
DISEASE_CLASSES = ["Petal Blight", "Brown Spot", "Soft Rot"]
ALL_CLASSES = HEALTHY_CLASSES + DISEASE_CLASSES

@st.cache_resource
def load_model():
    """Load YOLO model from .pt file"""
    try:
        model = YOLO("best.pt")
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None

def predict_disease_yolo(model, image):
    """Make prediction using YOLO model and return all detections."""
    if model is None: 
        return []
    
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
                
                for i in range(len(boxes)):
                    class_name = model.names[int(classes[i])]
                    
                    class_type = "healthy" if class_name in HEALTHY_CLASSES else "disease"
                    
                    detection = {
                        "disease": class_name,
                        "confidence": float(confidences[i]),
                        "box": boxes[i],
                        "class_type": class_type
                    }
                    detections.append(detection)
                    
                    # # DEBUGGING: Print untuk membantu debug
                    # print(f"Detected: {class_name} -> Type: {class_type}")
        
        return detections
        
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return []

def draw_detection_on_image(image, detections):
    """Draw all bounding boxes and labels on the image with different colors for healthy vs disease."""
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)

    cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    for detection in detections:
        box = detection["box"]
        disease = detection["disease"]
        confidence = detection["confidence"]
        class_type = detection["class_type"]
        
        x1, y1, x2, y2 = map(int, box)
        
        # Warna berbeda untuk healthy vs disease
        if class_type == "healthy":
            color = (0, 255, 0)  # Green for healthy
        else:
            color = (0, 0, 255)  # Red for disease
            
        cv2.rectangle(cv_image, (x1, y1), (x2, y2), color, 2)
        
        # Label yang lebih user-friendly
        if disease in ["healthy_leaf", "Healty Leaf", "Healthy Leaf"]:
            display_name = "Daun Sehat"
        elif disease in ["healthy_flower", "Healty Flower", "Healthy Flower"]:
            display_name = "Bunga Sehat"
        else:
            display_name = disease
            
        label = f"{display_name}: {confidence:.1%}"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(cv_image, (x1, y1 - label_size[1] - 10), (x1 + label_size[0], y1), color, -1)
        cv2.putText(cv_image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
    return cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

def analyze_detections(detections):
    """
    FIXED: Fungsi untuk menganalisis hasil deteksi dengan logika yang diperbaiki
    Returns: (status, message, diseases_found)
    """
    if not detections:
        return "no_plant", "Tidak ada tanaman anggrek terdeteksi dalam gambar", []
    
    # FIXED: Debug info
    print(f"Total detections: {len(detections)}")
    for det in detections:
        print(f"  - {det['disease']} ({det['class_type']})")
    
    # Pisahkan deteksi berdasarkan tipe
    healthy_detections = [d for d in detections if d["class_type"] == "healthy"]
    disease_detections = [d for d in detections if d["class_type"] == "disease"]
    
    print(f"Healthy detections: {len(healthy_detections)}")
    print(f"Disease detections: {len(disease_detections)}")
    
    if disease_detections:
        # Ada penyakit terdeteksi - ini yang prioritas
        disease_names = [d['disease'] for d in disease_detections]
        return "diseased", f"Terdeteksi {len(disease_detections)} area penyakit", disease_names
    
    elif healthy_detections:
        # FIXED: Hanya ada bagian sehat yang terdeteksi
        healthy_parts = []
        for d in healthy_detections:
            if d['disease'] in ["healthy_leaf", "Healty Leaf", "Healthy Leaf"]:
                healthy_parts.append('daun')
            elif d['disease'] in ["healthy_flower", "Healty Flower", "Healthy Flower"]:
                healthy_parts.append('bunga')
        
        healthy_parts_str = ', '.join(set(healthy_parts))
        return "healthy", f"Tanaman sehat - terdeteksi {healthy_parts_str} yang sehat", []
    
    else:
        # Tidak ada deteksi yang valid
        return "no_plant", "Tidak ada tanaman anggrek terdeteksi dalam gambar", []

def display_disease_info(disease_name):
    """Display disease information and recommendations."""
    if disease_name in DISEASE_INFO:
        info = DISEASE_INFO[disease_name]
        st.markdown(f"### 📋 Informasi & Rekomendasi untuk: {disease_name}")
        st.write(f"**Deskripsi:** {info['description']}")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("#### 🔍 Gejala Umum")
            for symptom in info['symptoms']: 
                st.write(f"• {symptom}")
        with col2:
            st.markdown("#### 🛡️ Cara Pencegahan")
            for prevention in info['prevention']: 
                st.write(f"• {prevention}")
        with col3:
            st.markdown("#### 💊 Cara Pengobatan")
            for treatment in info['treatment']: 
                st.write(f"• {treatment}")

def main():
    st.markdown('<h1 class="main-header">🌺 Orchid Disease Detection System</h1>', unsafe_allow_html=True)
    
    with st.sidebar:
        st.markdown("""
        <div class="sidebar-content">
            <h2>🎯 Fitur Aplikasi</h2>
            <p>Sistem AI untuk mendeteksi penyakit tanaman anggrek menggunakan model YOLO dengan 5 kelas deteksi.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### 📋 Kelas yang Dapat Dideteksi:")
        st.write("✅ Daun Sehat")
        st.write("🌸 Bunga Sehat") 
        st.write("🦠 Petal Blight")
        st.write("🍃 Brown Spot")
        st.write("🌿 Soft Rot")
        
        st.markdown("### 📊 Akurasi Model:")
        st.progress(0.89)
        st.write("Akurasi Rata-rata 89%")
        
        st.markdown("### 💡 Tips Penggunaan:")
        st.info("Gunakan foto dengan pencahayaan baik, fokus pada daun atau bunga anggrek, dan pastikan gambar tidak buram.")

    model = load_model()
    tab1, tab2 = st.tabs(["📷 Camera Capture", "📤 Upload Gambar"])
    
    def process_and_display_results(image):
        with st.spinner("Menganalisis gambar..."):
            detections = predict_disease_yolo(model, image)
            
            st.markdown("---")
            st.subheader("Hasil Analisis")
            
            col_res1, col_res2 = st.columns(2)
            with col_res1:
                st.image(image, caption="Gambar Asli", use_container_width=True)
            with col_res2:
                annotated_image = draw_detection_on_image(image, detections)
                st.image(annotated_image, caption="Hasil Deteksi AI", use_container_width=True)
            
            # FIXED: Analisis hasil dengan logika yang diperbaiki
            status, message, diseases_found = analyze_detections(detections)
            
            
            if status == "no_plant":
                st.markdown(f"""
                <div class="no-plant-result">
                    <h2>🔍 Tidak Ada Tanaman Terdeteksi</h2>
                    <p>{message}</p>
                    <p>Pastikan gambar menampilkan daun atau bunga anggrek dengan jelas.</p>
                </div>
                """, unsafe_allow_html=True)
                
            elif status == "healthy":
                st.markdown(f"""
                <div class="healthy-result">
                    <h2>✅ Tanaman Sehat!</h2>
                    <p>{message}</p>
                    <p>Tidak ditemukan tanda-tanda penyakit pada tanaman anggrek Anda.</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Tampilkan tips perawatan untuk tanaman sehat
                st.markdown("""
                <div class="recommendation-card">
                    <h3>🌱 Tips Perawatan Lanjutan:</h3>
                    <ul>
                        <li>Pertahankan kelembaban udara 50-70%</li>
                        <li>Berikan cahaya tidak langsung yang cukup</li>
                        <li>Siram secukupnya, jangan berlebihan</li>
                        <li>Lakukan pemupukan rutin sebulan sekali</li>
                        <li>Periksa tanaman secara berkala untuk deteksi dini penyakit</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
                
            elif status == "diseased":
                most_common_disease = Counter(diseases_found).most_common(1)[0][0]
                unique_diseases = list(set(diseases_found))
                
                st.markdown(f"""
                <div class="detection-result">
                    <h2>⚠️ Penyakit Terdeteksi!</h2>
                    <p>{message}</p>
                    <p>Jenis penyakit: <strong>{', '.join(unique_diseases)}</strong></p>
                    <p>Segera lakukan tindakan pengobatan untuk mencegah penyebaran!</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Tampilkan informasi detail penyakit yang paling umum
                display_disease_info(most_common_disease)

    with tab1:
        st.markdown("""
        <div class="feature-card">
            <h3>📷 Deteksi dengan Camera Capture</h3>
            <p>Ambil foto menggunakan kamera untuk analisis penyakit secara langsung.</p>
        </div>
        """, unsafe_allow_html=True)
        
        if not st.session_state.camera_activated:
            if st.button("📷 Aktifkan Kamera"):
                st.session_state.camera_activated = True
                st.rerun()
        else:
            st.info("Kamera aktif. Silakan posisikan daun atau bunga anggrek dan ambil foto.")
            camera_input = st.camera_input("Arahkan kamera ke tanaman anggrek...", key="camera", label_visibility="collapsed")
            
            if st.button("❌ Matikan Kamera"):
                st.session_state.camera_activated = False
                st.rerun()

            if camera_input is not None:
                image = Image.open(camera_input)
                process_and_display_results(image)
    
    with tab2:
        st.markdown("""
        <div class="feature-card">
            <h3>📤 Upload Gambar</h3>
            <p>Upload foto tanaman anggrek dari galeri Anda untuk dianalisis.</p>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader("Pilih gambar tanaman anggrek", type=['jpg', 'jpeg', 'png'])
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            
            if st.button("🔍 Analisis Penyakit", key="upload_analyze"):
                process_and_display_results(image)
    
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 2rem;">
        <p>🌺 Orchid Disease Detection System | Powered by AI & YOLO</p>
        <p>Developed with ❤️ for orchid enthusiasts</p>
        <p>Model dapat mendeteksi: Daun Sehat, Bunga Sehat, dan 3 Jenis Penyakit</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()