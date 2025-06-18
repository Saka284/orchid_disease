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

# CSS styling
st.markdown("""
<style>
    /* KODE ASLI ANDA, TIDAK DIUBAH */
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
    
    /* MENGHAPUS SEMUA CSS TOMBOL LAMA DAN MENGGANTINYA DENGAN YANG BARU */
    .info-card-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 20px;
        margin-top: 1.5rem;
    }
    .info-card, .info-card-treatment {
        background-color: #2C3E50;
        color: #ECF0F1;
        padding: 25px;
        border-radius: 15px;
        border: 1px solid #34495E;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        display: flex;
        flex-direction: column;
    }
    .info-card-treatment {
        grid-column: 1 / -1;
        margin-top: 20px;
        background-color: #233140; 
    }
    .info-card h4, .info-card-treatment h4 {
        font-size: 1.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 20px;
        padding-bottom: 15px;
        color: #5DADE2; 
        border-bottom: 1px solid #34495E;
        text-shadow: none;
    }
    .info-card ul, .info-card-treatment ul {
        list-style-type: none;
        padding-left: 0;
        flex-grow: 1;
    }
    .info-card li, .info-card-treatment li {
        background-color: rgba(52, 73, 94, 0.5); 
        padding: 12px;
        border-radius: 10px;
        margin-bottom: 10px;
        font-size: 0.95rem;
        line-height: 1.4;
        border-left: 3px solid #5DADE2;
    }

    /* === CSS TOMBOL FINAL YANG PASTI BERHASIL === */
    .tombol-merah .stButton,
    .tombol-outline .stButton {
        width: 100%;
    }
    .tombol-merah button,
    .tombol-outline button {
        width: 100%;
        font-weight: bold;
        border-radius: 10px;
        padding: 0.75rem 1rem;
        transition: all 0.2s ease-in-out;
    }
    /* Tombol Aksi Utama (MERAH) */
    .tombol-merah button {
        background-color: #FF6B6B; /* Warna merah persis dari header Anda */
        color: white;
        border: none;
    }
    .tombol-merah button:hover {
        background-color: #E55353; /* Warna merah lebih gelap saat disentuh */
        transform: translateY(-2px);
    }
    /* Tombol Aksi Sekunder (Outline) */
    .tombol-outline button {
        background-color: transparent;
        color: #F5B7B1; /* Warna teks merah muda */
        border: 2px solid #FF6B6B; /* Outline warna merah */
    }
    .tombol-outline button:hover {
        background-color: #FF6B6B;
        color: white;
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
        # Ganti "best.pt" dengan path ke file model Anda
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
        st.markdown("""
        <div class="feature-card">
            <h3>Selamat Datang di Asisten Kesehatan Anggrek Anda!</h3>
            <p>Aplikasi ini menggunakan Kecerdasan Buatan (AI) untuk membantu Anda mengidentifikasi penyakit umum pada anggrek secara cepat dan akurat. Deteksi dini adalah kunci untuk menyelamatkan tanaman kesayangan Anda.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.subheader("Langkah-langkah Penggunaan")
        st.markdown("""
        1.  **Pilih Metode Input**:
            - Buka tab **Camera Capture** untuk mengambil foto anggrek Anda secara langsung.
            - Buka tab **Upload Image** untuk menganalisis foto dari galeri perangkat Anda.
        
        2.  **Ambil atau Unggah Gambar**:
            - Pastikan gambar yang Anda ambil memiliki **pencahayaan yang baik** dan **fokus yang tajam** pada bagian tanaman (bunga, daun, atau batang) yang diduga sakit.
            
        3.  **Dapatkan Hasil & Solusi**:
            - Tekan tombol analisis dan biarkan sistem kami bekerja.
            - Anda akan mendapatkan hasil deteksi penyakit beserta informasi lengkap mengenai gejala, cara pencegahan, dan metode pengobatan yang direkomendasikan.
        """)

        st.info("💡 **Tips**: Semakin jelas gambar Anda, semakin akurat hasilnya. Silakan pilih salah satu tab di atas untuk memulai!")


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
            # --- TOMBOL PRIMER DENGAN WRAPPER ---
            st.markdown('<div class="tombol-merah">', unsafe_allow_html=True)
            if st.button("📷 Activate Camera", key="activate_camera"):
                st.session_state.camera_activated = True
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.info("Camera is active. Please position the orchid's leaf or flower and take a picture.")
            camera_input = st.camera_input("Point the camera at the orchid plant...", key="camera", label_visibility="collapsed")

            # --- TOMBOL SEKUNDER DENGAN WRAPPER ---
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
            st.image(image, caption="Image to be analyzed", use_container_width=True)

            # --- TOMBOL PRIMER DENGAN WRAPPER ---
            st.markdown('<div class="tombol-merah">', unsafe_allow_html=True)
            if st.button("🔍 Analyze Disease", key="upload_analyze"):
                process_and_display_results(image)
            st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 2rem;">
        <p>🌺 Orchid Disease Detection System | Powered by AI & YOLO</p>
        <p>Developed with ❤️ for orchid enthusiasts</p>
        <p>The model can detect 3 Types of Diseases: Petal Blight, Brown Spot, and Soft Rot</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()