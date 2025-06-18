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

# Initialize session_state for camera control
if 'camera_activated' not in st.session_state:
    st.session_state.camera_activated = False

# --- CSS FINAL YANG SUDAH DIBERSIHKAN ---
st.markdown("""
<style>
    /* BAGIAN 1: SEMUA GAYA ASLI ANDA (TIDAK DIUBAH) */
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
    .feature-card:hover { transform: translateY(-5px); }
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
    .info-card-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 20px;
        margin-top: 1.5rem;
    }
    .info-card {
        background: linear-gradient(145deg, #ffc371, #ff5f6d);
        border-radius: 20px;
        padding: 25px;
        color: white;
        box-shadow: 0 10px 25px rgba(0,0,0,0.25);
        display: flex;
        flex-direction: column;
    }
    .info-card h4 {
        font-size: 1.5rem; font-weight: bold; color: white; text-align: center;
        margin-bottom: 20px; border-bottom: 1px solid rgba(255, 255, 255, 0.3); padding-bottom: 15px;
    }
    .info-card ul { list-style-type: none; padding-left: 0; flex-grow: 1; }
    .info-card li {
        background-color: rgba(0, 0, 0, 0.15); padding: 12px;
        border-radius: 10px; margin-bottom: 10px; font-size: 0.95rem; line-height: 1.4;
    }
    .info-card-treatment {
        background: linear-gradient(145deg, #84fab0, #8fd3f4);
        grid-column: 1 / -1; border-radius: 20px; padding: 25px; color: #1f3b4d;
        box-shadow: 0 10px 25px rgba(0,0,0,0.25); margin-top: 20px;
    }
    .info-card-treatment h4 {
        color: #1f3b4d; border-bottom: 1px solid rgba(31, 59, 77, 0.3);
    }
    .info-card-treatment li { background-color: rgba(255, 255, 255, 0.4); }

    /* BAGIAN 2: CSS BARU UNTUK UI BERANDA */
    .home-card {
        background-color: #2C3E50; padding: 2rem; border-radius: 15px;
        border: 1px solid #34495E; text-align: center; height: 100%; color: #ECF0F1;
    }
    .home-card h3 { color: #5DADE2; margin-bottom: 1rem; }
    .dos-donts-list { list-style-type: none; padding-left: 0; color: #ECF0F1; text-align: left; }
    .dos-donts-list li { margin-bottom: 0.5rem; padding: 0.5rem; border-radius: 7px; }
    .dos { background-color: rgba(46, 204, 113, 0.1); border-left: 4px solid #2ECC71; }
    .donts { background-color: rgba(231, 76, 60, 0.1); border-left: 4px solid #E74C3C; }

    /* BAGIAN 3: GAYA TOMBOL ASLI ANDA SEBAGAI DEFAULT */
    .stButton > button {
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4);
        color: white; border: none; border-radius: 25px; padding: 0.5rem 2rem;
        font-weight: bold; transition: all 0.3s ease; width: 100%;
    }
    .stButton > button:hover {
        transform: scale(1.05); box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }

    /* BAGIAN 4: CSS KHUSUS UNTUK MENIMPA GAYA TOMBOL TERTENTU */
    .tombol-merah button {
        background: #FF6B6B !important;
        border-radius: 10px !important;
    }
    .tombol-merah button:hover {
        background: #E55353 !important;
        transform: translateY(-2px) !important;
    }
    .tombol-outline button {
        background: transparent !important;
        color: #FF6B6B !important;
        border: 2px solid #FF6B6B !important;
        border-radius: 10px !important;
    }
    .tombol-outline button:hover {
        background: #FF6B6B !important;
        color: white !important;
        transform: scale(1.05) !important;
    }
</style>
""", unsafe_allow_html=True)

# Disease information database
DISEASE_INFO = {
    "Petal Blight": {
        "description": "A petal blight disease caused by the fungus Botrytis cinerea.",
        "symptoms": ["Brown spots on flower petals", "Petals become soft and slimy", "Premature flower drop"],
        "prevention": ["Maintain good air circulation", "Avoid excess humidity", "Remove faded flowers promptly"],
        "treatment": ["Cut off all infected parts", "Apply a suitable fungicide", "Reduce watering and humidity"]
    },
    "Brown Spot": {
        "description": "A brown rot caused by the fungus Monilinia fructicola.",
        "symptoms": ["Brown spots on the pseudobulb", "Tissue becomes soft and rotten", "Appearance of brown-colored spores"],
        "prevention": ["Keep the area clean", "Avoid injuring the pseudobulb", "Ensure good drainage"],
        "treatment": ["Cut infected tissue down to healthy areas", "Apply fungicide paste to the wound", "Isolate the plant"]
    },
    "Soft Rot": {
        "description": "A soft rot caused by the bacterium Erwinia carotovora.",
        "symptoms": ["Plant tissue becomes mushy", "A foul, rotting smell", "The infected part changes color"],
        "prevention": ["Avoid overwatering", "Keep tools and medium clean", "Ensure good air circulation"],
        "treatment": ["Cut off all infected parts", "Apply a bactericide like streptomycin", "Reduce humidity"]
    }
}

# Only disease classes are now relevant
DISEASE_CLASSES = ["Petal Blight", "Brown Spot", "Soft Rot"]

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
    """Make prediction using YOLO model and return only detections of specified diseases."""
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

                    if class_name in DISEASE_CLASSES:
                        detection = {
                            "disease": class_name,
                            "confidence": float(confidences[i]),
                            "box": boxes[i],
                        }
                        detections.append(detection)

        return detections

    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return []

def draw_detection_on_image(image, detections):
    """Draw all bounding boxes and labels on the image. All detections are red."""
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)

    cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    for detection in detections:
        box = detection["box"]
        disease = detection["disease"]
        confidence = detection["confidence"]

        x1, y1, x2, y2 = map(int, box)
        color = (0, 0, 255)  # Red for disease

        cv2.rectangle(cv_image, (x1, y1), (x2, y2), color, 2)

        label = f"{disease}: {confidence:.1%}"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(cv_image, (x1, y1 - label_size[1] - 10), (x1 + label_size[0], y1), color, -1)
        cv2.putText(cv_image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

def analyze_detections(detections):
    """Simplified analysis. If there are any detections, it's a disease. Otherwise, no disease is detected."""
    if not detections:
        return "no_disease_detected", "No disease was detected on the plant.", []
    else:
        disease_names = [d['disease'] for d in detections]
        return "diseased", f"Detected {len(disease_names)} area(s) of disease.", disease_names

def display_disease_info(disease_name):
    """Display disease information and recommendations in a colorful, two-above-one card layout."""
    if disease_name in DISEASE_INFO:
        info = DISEASE_INFO.get(disease_name)
        if not info:
            return

        st.markdown(f"### 📋 Information & Recommendations for: {disease_name}")

        symptoms_list = ''.join([f"<li>{symptom}</li>" for symptom in info['symptoms']])
        prevention_list = ''.join([f"<li>{prevention}</li>" for prevention in info['prevention']])
        treatment_list = ''.join([f"<li>{treatment}</li>" for treatment in info['treatment']])

        card_html = f"""
        <div class="info-card-grid">
            <div class="info-card">
                <h4>🔍 Common Symptoms</h4>
                <ul>{symptoms_list}</ul>
            </div>
            <div class="info-card">
                <h4>🛡️ Prevention Methods</h4>
                <ul>{prevention_list}</ul>
            </div>
            <div class="info-card-treatment">
                <h4>💊 Treatment Methods</h4>
                <ul>{treatment_list}</ul>
            </div>
        </div>
        """
        st.markdown(card_html, unsafe_allow_html=True)

def main():
    st.markdown('<h1 class="main-header">🌺 Orchid Disease Detection System</h1>', unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("""
        <div class="sidebar-content">
            <h2>🎯 Application Features</h2>
            <p>An AI system to detect specific diseases on orchid plants using a YOLO model.</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### 📋 Detectable Diseases:")
        st.write("🦠 Petal Blight")
        st.write("🍃 Brown Spot")
        st.write("🌿 Soft Rot")

        st.markdown("### 📊 Model Accuracy:")
        st.progress(0.89)
        st.write("Average Accuracy: 89%")

        st.markdown("### 💡 Usage Tips:")
        st.info("Use a photo with good lighting, focus on the orchid's leaf or flower, and ensure the image is not blurry.")

    model = load_model()
    
    tab_beranda, tab_camera, tab_upload = st.tabs(["🏠 Beranda", "📷 Camera Capture", "📤 Upload Image"])

    with tab_beranda:
        st.subheader("Membantu Anda Merawat Anggrek dengan Kecerdasan Buatan")
        st.markdown(
            "Selamat datang! Aplikasi ini dirancang untuk menjadi asisten pribadi Anda dalam menjaga kesehatan anggrek. "
            "Gunakan kekuatan AI untuk mendeteksi penyakit secara dini dan dapatkan rekomendasi perawatan yang tepat."
        )
        st.markdown("---")

        st.header("🔍 Penyakit yang Dapat Dideteksi")
        st.info("Model kami saat ini dilatih untuk mengenali 3 penyakit umum pada **daun dan bunga** anggrek.")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(
                """<div class="home-card"><h3>Petal Blight</h3><p>Bercak coklat atau kehitaman pada kelopak bunga yang membuatnya cepat layu.</p></div>""",
                unsafe_allow_html=True
            )
        with col2:
            st.markdown(
                """<div class="home-card"><h3>Brown Spot</h3><p>Bercak coklat pada daun yang bisa menyebar dan menyebabkan pembusukan jaringan.</p></div>""",
                unsafe_allow_html=True
            )
        with col3:
            st.markdown(
                """<div class="home-card"><h3>Soft Rot</h3><p>Pembusukan lunak dan berbau pada daun atau batang, seringkali akibat bakteri.</p></div>""",
                unsafe_allow_html=True
            )

        st.markdown("---")

        st.header("✅ Praktik Terbaik untuk Hasil Akurat")
        col_dos, col_donts = st.columns(2)
        with col_dos:
            st.markdown(
                """
                <div style="text-align:left;"><h4><span style="color:#2ECC71;">Lakukan Ini:</span></h4></div>
                <ul class="dos-donts-list">
                    <li class="dos">Gunakan pencahayaan yang cukup & merata.</li>
                    <li class="dos">Fokuskan foto pada satu area (daun/bunga).</li>
                    <li class="dos">Pastikan gambar jelas dan tidak buram.</li>
                    <li class="dos">Ambil foto dengan latar belakang polos jika memungkinkan.</li>
                </ul>
                """,
                unsafe_allow_html=True
            )
        with col_donts:
            st.markdown(
                """
                <div style="text-align:left;"><h4><span style="color:#E74C3C;">Hindari Ini:</span></h4></div>
                <ul class="dos-donts-list">
                    <li class="donts">Mengambil foto di tempat yang terlalu gelap/terang.</li>
                    <li class="donts">Menyertakan terlalu banyak bagian tanaman dalam satu foto.</li>
                    <li class="donts">Menggunakan gambar yang pecah atau buram.</li>
                    <li class="donts">Bayangan yang menutupi area yang sakit.</li>
                </ul>
                """,
                unsafe_allow_html=True
            )
            
        st.markdown("---")

        st.warning(
            "**Penting:** Aplikasi ini adalah alat bantu deteksi dan bukan pengganti diagnosis dari ahli hortikultura profesional. "
            "Hasil deteksi memiliki tingkat akurasi tertentu dan harus digunakan sebagai panduan awal."
        )


    def process_and_display_results(image):
        with st.spinner("Analyzing the image..."):
            detections = predict_disease_yolo(model, image)

            st.markdown("---")
            st.subheader("Analysis Results")

            col_res1, col_res2 = st.columns(2)
            with col_res1:
                st.image(image, caption="Original Image", use_container_width=True)
            with col_res2:
                annotated_image = draw_detection_on_image(image, detections)
                st.image(annotated_image, caption="AI Detection Result", use_container_width=True)

            status, message, diseases_found = analyze_detections(detections)

            if status == "no_disease_detected":
                st.markdown(f"""
                <div class="healthy-result">
                    <h2>✅ No Disease Detected!</h2>
                    <p>{message}</p>
                    <p>Your orchid appears to be free from the detected diseases. Keep up the great care!</p>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("""
                <div class="recommendation-card">
                    <h3>🌱 General Orchid Care Recommendations:</h3>
                    <ul>
                        <li><strong>Light:</strong> Provide bright, indirect sunlight. Avoid direct sun which can scorch leaves.</li>
                        <li><strong>Watering:</strong> Water thoroughly when the growing medium is almost dry. Do not let it sit in water.</li>
                        <li><strong>Humidity:</strong> Orchids thrive in 50-70% humidity. Consider a humidifier or a pebble tray.</li>
                        <li><strong>Airflow:</strong> Good air circulation is crucial to prevent fungal and bacterial issues.</li>
                        <li><strong>Fertilizing:</strong> Use a balanced orchid fertilizer weakly, weekly during the growing season.</li>
                        <li><strong>Inspect Regularly:</strong> Check your plant often for any early signs of pests or disease.</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)

            elif status == "diseased":
                most_common_disease = Counter(diseases_found).most_common(1)[0][0]
                unique_diseases = list(set(diseases_found))

                st.markdown(f"""
                <div class="detection-result">
                    <h2>⚠️ Disease Detected!</h2>
                    <p>{message}</p>
                    <p>Disease types: <strong>{', '.join(unique_diseases)}</strong></p>
                    <p>Take immediate action to prevent the disease from spreading!</p>
                </div>
                """, unsafe_allow_html=True)

                display_disease_info(most_common_disease)

    with tab_camera:
        st.markdown("""
        <div class="feature-card">
            <h3>📷 Detect with Camera Capture</h3>
            <p>Take a photo using your camera for instant disease analysis.</p>
        </div>
        """, unsafe_allow_html=True)

        if not st.session_state.camera_activated:
            st.markdown('<div class="tombol-merah">', unsafe_allow_html=True)
            if st.button("📷 Activate Camera", key="activate_camera"):
                st.session_state.camera_activated = True
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.info("Camera is active. Please position the orchid's leaf or flower and take a picture.")
            camera_input = st.camera_input("Point the camera at the orchid plant...", key="camera", label_visibility="collapsed")

            st.markdown('<div class="tombol-outline">', unsafe_allow_html=True)
            if st.button("❌ Deactivate Camera", key="deactivate_camera"):
                st.session_state.camera_activated = False
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

            if camera_input is not None:
                image = Image.open(camera_input)
                process_and_display_results(image)

    with tab_upload:
        st.markdown("""
        <div class="feature-card">
            <h3>📤 Upload an Image</h3>
            <p>Upload a photo of your orchid plant from your gallery for analysis.</p>
        </div>
        """, unsafe_allow_html=True)

        uploaded_file = st.file_uploader("Choose an orchid image", type=['jpg', 'jpeg', 'png'])

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            
            st.markdown('<div class="tombol-merah">', unsafe_allow_html=True)
            if st.button("🔍 Analyze Disease", key="upload_analyze"):
                process_and_display_results(image)
            st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 2rem;">
        <p>🌺 Orchid Disease Detection System | Powered by AI & YOLO</p>
        <p>Developed with ❤️ for orchid enthusiasts</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()