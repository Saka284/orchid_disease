import streamlit as st
import cv2
import numpy as np
from PIL import Image
from collections import Counter

from ultralytics import YOLO

# ... (CSS Anda tetap sama, tidak perlu diubah) ...
st.markdown("""
<style>
    /* ... (Semua CSS Anda ada di sini) ... */

    /* Menambahkan beberapa style untuk halaman selamat datang */
    .welcome-section {
        text-align: center;
        padding: 2rem;
    }
    .welcome-icon {
        font-size: 5rem;
        margin-bottom: 1rem;
    }
    .step-card {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 5px solid #FF6B6B;
        text-align: left;
    }
    .step-card h4 {
        color: #FF6B6B;
    }
</style>
""", unsafe_allow_html=True)


# ... (DISEASE_INFO dan DISEASE_CLASSES tetap sama) ...
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

DISEASE_CLASSES = ["Petal Blight", "Brown Spot", "Soft Rot"]


# ... (Semua fungsi Anda seperti load_model, predict_disease_yolo, dll. tetap sama) ...
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

        # Renders the 2x1 top grid and the full-width bottom card
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
        # ... (Isi sidebar Anda tetap sama) ...
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
    
    # --- PERUBAHAN UTAMA: Menambahkan tab 'Selamat Datang' ---
    tab0, tab1, tab2 = st.tabs(["👋 Selamat Datang", "📷 Kamera", "📤 Unggah Gambar"])

    with tab0:
        st.markdown('<div class="welcome-section">', unsafe_allow_html=True)
        st.markdown('<div class="welcome-icon">🌸</div>', unsafe_allow_html=True)
        st.markdown('<h2>Selamat Datang di Sistem Deteksi Penyakit Anggrek!</h2>', unsafe_allow_html=True)
        
        st.markdown("""
        <p style="font-size: 1.1rem; max-width: 800px; margin: auto;">
            Aplikasi ini dirancang untuk membantu para pecinta anggrek seperti Anda. Dengan teknologi AI canggih, 
            Anda dapat mendeteksi penyakit umum pada anggrek secara dini, memberikan kesempatan terbaik bagi 
            tanaman Anda untuk pulih dan tumbuh subur.
        </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        st.subheader("🚀 Cara Memulai dalam 3 Langkah Mudah")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""
            <div class="step-card">
                <h4>1. Pilih Sumber Gambar</h4>
                <p>Buka tab <b>'📷 Kamera'</b> untuk mengambil foto langsung atau tab <b>'📤 Unggah Gambar'</b> untuk memilih foto dari perangkat Anda.</p>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown("""
            <div class="step-card">
                <h4>2. Siapkan Gambar</h4>
                <p>Pastikan gambar memiliki pencahayaan yang baik, fokus, dan jelas menyorot bagian bunga, daun, atau batang yang ingin diperiksa.</p>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown("""
            <div class="step-card">
                <h4>3. Analisis & Dapatkan Solusi</h4>
                <p>Klik tombol analisis dan biarkan AI bekerja. Anda akan menerima hasil deteksi beserta informasi gejala dan rekomendasi perawatan.</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        st.info("💡 **Tips Pro:** Semakin jelas gambar Anda, semakin akurat hasil deteksi yang akan Anda dapatkan. Silakan mulai dengan memilih salah satu tab di atas!")


    def process_and_display_results(image):
        # ... (Fungsi ini tetap sama) ...
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
        

    with tab1:
        # ... (Isi tab1 Anda tetap sama) ...
        st.markdown("""
        <div class="feature-card">
            <h3>📷 Detect with Camera Capture</h3>
            <p>Take a photo using your camera for instant disease analysis.</p>
        </div>
        """, unsafe_allow_html=True)

        if 'camera_activated' not in st.session_state:
            st.session_state.camera_activated = False

        if not st.session_state.camera_activated:
            if st.button("📷 Activate Camera"):
                st.session_state.camera_activated = True
                st.rerun()
        else:
            st.info("Camera is active. Please position the orchid's leaf or flower and take a picture.")
            camera_input = st.camera_input("Point the camera at the orchid plant...", key="camera", label_visibility="collapsed")

            if st.button("❌ Deactivate Camera"):
                st.session_state.camera_activated = False
                st.rerun()

            if camera_input is not None:
                image = Image.open(camera_input)
                process_and_display_results(image)
        

    with tab2:
        # ... (Isi tab2 Anda tetap sama) ...
        st.markdown("""
        <div class="feature-card">
            <h3>📤 Upload an Image</h3>
            <p>Upload a photo of your orchid plant from your gallery for analysis.</p>
        </div>
        """, unsafe_allow_html=True)

        uploaded_file = st.file_uploader("Choose an orchid image", type=['jpg', 'jpeg', 'png'])

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            
            # Menampilkan gambar yang diunggah sebelum tombol analisis ditekan
            st.image(image, caption="Image to be analyzed.", use_container_width=True)

            if st.button("🔍 Analyze Disease", key="upload_analyze"):
                process_and_display_results(image)

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