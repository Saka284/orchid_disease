import streamlit as st
import cv2
import numpy as np
from PIL import Image
from collections import Counter
from ultralytics import YOLO

# Page Config
st.set_page_config(
    page_title="🌺 Orchid Disease Detection",
    page_icon="🌺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session_state for camera control
if 'camera_activated' not in st.session_state:
    st.session_state.camera_activated = False

# Enhanced CSS with modern design and responsiveness
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
# DISEASE DATABASE - ENGLISH VERSION
# ==============================================================================
DISEASE_INFO = {
    "Petal Blight": {
        "description": "Petal blight is caused by fungi like Botrytis cinerea (grey mold) or Phytophthora. This disease attacks buds and flowers, causing significant losses.",
        "symptoms": [
            "Light brown, watery lesions on flower petals.",
            "In Botrytis infections, grey, dust-like spores may appear.",
            "Phytophthora infection does not produce grey spores but still causes a wet rot.",
            "Flower buds may rot and fail to open."
        ],
        "prevention": [
            "Remove faded or infected flowers promptly.",
            "Increase air circulation around the flowers to reduce humidity.",
            "Avoid spraying water directly onto the flowers.",
            "Keep the growing area clean of plant debris."
        ],
        "treatment": [
            "Use a fungicide effective against Botrytis or Phytophthora (e.g., Captan, Aliette, Subdue).",
            "Apply preventive sprays if environmental conditions are very humid.",
            "Cut and destroy all infected parts to stop the spread.",
            "Consider biological control agents (antagonistic microorganisms) if available."
        ]
    },
    "Brown Spot": {
        "description": "Brown Spot or Brown Rot is a destructive disease caused by fungi (like Phytophthora) or bacteria (like Erwinia), attacking leaves and pseudobulbs.",
        "symptoms": [
            "Water-logged spots on leaves, initially yellowish-brown.",
            "Spots rapidly enlarge and turn dark brown or black.",
            "In some orchid species, the infection starts at the base of the leaf and spreads upwards.",
            "In severe cases, it can spread to the roots, causing root rot."
        ],
        "prevention": [
            "Maintain good air circulation to reduce humidity.",
            "Avoid leaving leaves wet for extended periods; do not water from above.",
            "Ensure the potting medium has good drainage.",
            "Always use sterile cutting tools for maintenance."
        ],
        "treatment": [
            "Immediately cut off infected plant parts into healthy tissue using a sterile tool.",
            "Apply a fungicide/bactericide paste (e.g., Physan 20, Captan, Aliette) to the wound.",
            "For Phytophthora, systemic fungicides like Aliette or Subdue are very effective.",
            "Isolate the sick plant to prevent transmission."
        ]
    },
    "Soft Rot": {
        "description": "Soft Rot is a highly dangerous and fast-spreading bacterial disease caused by Pectobacterium or Dickeya. This disease is often fatal, especially for Phalaenopsis.",
        "symptoms": [
            "Leaves become translucent, wet, and mushy.",
            "Emits a characteristic foul, rotting smell.",
            "Often starts as a small, water-soaked spot surrounded by a yellow halo.",
            "Spreads very quickly, can destroy an entire plant in a matter of days."
        ],
        "prevention": [
            "Keep leaves dry at all times. Water only the potting medium.",
            "Maximize air circulation around the plant.",
            "Avoid mechanical injuries to leaves and roots, which can be entry points for bacteria.",
            "Inspect plants regularly, especially during warm, humid weather."
        ],
        "treatment": [
            "This is an emergency! Immediately cut off all infected parts well into the healthy tissue.",
            "Use a blade sterilized with a flame or alcohol for EVERY cut.",
            "Apply a copper-based bactericide/fungicide powder or an antibiotic to the wound.",
            "Withhold watering temporarily and isolate the plant from others."
        ]
    }
}

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
    """Make prediction using YOLO model"""
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
                    class_id = int(classes[i])
                    class_name = model.names[class_id]
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
    """Draw detection boxes on the image."""
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)

    cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    for detection in detections:
        box = detection["box"]
        disease = detection["disease"]
        confidence = detection["confidence"]
        x1, y1, x2, y2 = map(int, box)
        color = (0, 0, 255)  # Red for disease

        cv2.rectangle(cv_image, (x1, y1), (x2, y2), color, 3)
        label = f"{disease}: {confidence:.1%}"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(cv_image, (x1, y1 - label_size[1] - 15), (x1 + label_size[0] + 10, y1), color, -1)
        cv2.putText(cv_image, label, (x1 + 5, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

def analyze_detections(detections):
    """Analyze detections to determine the status."""
    if not detections:
        return "no_disease_detected", "No disease was detected on the plant.", []
    else:
        disease_names = [d['disease'] for d in detections]
        return "diseased", f"Detected {len(disease_names)} area(s) of disease.", disease_names

def display_disease_info(disease_name):
    """Display disease information and recommendations."""
    info = DISEASE_INFO.get(disease_name)
    if not info:
        return

    st.markdown(f"### 📋 Information & Recommendations for: **{disease_name}**")
    symptoms_list = ''.join([f"<li>{symptom}</li>" for symptom in info['symptoms']])
    prevention_list = ''.join([f"<li>{prevention}</li>" for prevention in info['prevention']])
    treatment_list = ''.join([f"<li>{treatment}</li>" for treatment in info['treatment']])

    card_html = f"""
    <div class="info-card-grid">
        <div class="info-card"><h4>🔍 Common Symptoms</h4><ul>{symptoms_list}</ul></div>
        <div class="info-card"><h4>🛡️ Prevention Methods</h4><ul>{prevention_list}</ul></div>
        <div class="info-card-treatment"><h4>💊 Treatment Methods</h4><ul>{treatment_list}</ul></div>
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)

def process_and_display_results(image, model):
    """Process image, run prediction, and display results."""
    with st.spinner("🔍 Analyzing your orchid image..."):
        detections = predict_disease_yolo(model, image)

    st.markdown("---")
    st.subheader("📊 Analysis Results")

    col_res1, col_res2 = st.columns(2)
    with col_res1:
        st.image(image, caption="📷 Original Image", use_container_width=True)
    with col_res2:
        annotated_image = draw_detection_on_image(image, detections)
        st.image(annotated_image, caption="🤖 AI Detection Result", use_container_width=True)

    status, message, diseases_found = analyze_detections(detections)

    if status == "no_disease_detected":
        st.markdown(f"""
        <div class="healthy-result">
            <h2>✅ Excellent News!</h2><h3>No Disease Detected</h3>
            <p style="font-size: 1.1rem; margin: 1rem 0;">{message}</p>
            <p>Your orchid appears to be healthy. Keep up the great care! 🌟</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div class="recommendation-card">
            <h4>🌱 Healthy Orchid Care Tips</h4>
            <ul>
                <li><strong>💡 Lighting:</strong> Bright, indirect sunlight. East-facing windows are ideal.</li>
                <li><strong>💧 Watering:</strong> Water thoroughly when the potting medium is almost dry. Avoid waterlogging.</li>
                <li><strong>🌡️ Humidity:</strong> Orchids thrive in 50-70% humidity. Consider a humidifier or a pebble tray.</li>
                <li><strong>🌬️ Airflow:</strong> Good air circulation is crucial to prevent fungal and bacterial issues.</li>
                <li><strong>🌿 Fertilizing:</strong> Use a balanced orchid fertilizer weakly during the growing season.</li>
                <li><strong>🧐 Regular Inspection:</strong> Check your plant often for any early signs of pests or disease.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    elif status == "diseased":
        most_common_disease = Counter(diseases_found).most_common(1)[0][0]
        unique_diseases = list(set(diseases_found))
        st.markdown(f"""
        <div class="detection-result">
            <h2>⚠️ Disease Detected!</h2><p>{message}</p>
            <p>Disease types: <strong>{', '.join(unique_diseases)}</strong></p>
            <p>Take immediate action to prevent the disease from spreading!</p>
        </div>
        """, unsafe_allow_html=True)
        display_disease_info(most_common_disease)

def main():
    st.markdown('<h1 class="main-header">🌺 Orchid Disease Detection System</h1>', unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("""
        <div class="sidebar-content">
            <h2>🎯 Application Features</h2>
            <p>An AI-powered system to detect specific diseases on orchid plants using advanced YOLO technology.</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("### 🦠 Detectable Diseases:")
        diseases = [
            ("🌸 Petal Blight", "Fungal infection affecting flowers."),
            ("🍃 Brown Spot", "Rot affecting leaves & pseudobulbs."),
            ("🌿 Soft Rot", "A fatal bacterial soft rot.")
        ]
        for disease, desc in diseases:
            st.markdown(f"**{disease}**")
            st.caption(desc)
        st.markdown("### 📊 Model Performance:")
        st.progress(0.89)
        st.markdown("**Average Accuracy: 89%**")
        st.caption("Trained on 10,000+ orchid images")
        st.markdown("### 💡 Pro Tips:")
        st.info("🔍 Use good lighting\n\n📸 Focus on affected areas\n\n🎯 Avoid blurry images\n\n🌟 Plain backgrounds work best")

    model = load_model()
    
    tab_beranda, tab_camera, tab_upload = st.tabs(["🏠 Home", "📷 Camera", "📤 Upload"])

    with tab_beranda:
        st.markdown("""
        <div style="text-align: center; margin: 2rem 0;">
            <h2 style="color: #5dade2; margin-bottom: 1rem;">🤖 AI-Powered Orchid Care Assistant</h2>
            <p style="font-size: 1.1rem; color: #b3b3b3; max-width: 800px; margin: 0 auto;">
                Welcome to the future of orchid care! Our advanced AI system helps you identify diseases early 
                and provides expert recommendations to keep your orchids healthy and thriving.
            </p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("---")
        st.markdown("## 🎯 Detectable Diseases")
        st.markdown("""
        <div class="disease-grid">
            <div class="disease-card"><div class="disease-icon">🌸</div><h4>Petal Blight</h4><p>Attacks buds and flowers, causing wet spots and rot, often caused by the Botrytis fungus.</p></div>
            <div class="disease-card"><div class="disease-icon">🍃</div><h4>Brown Spot</h4><p>Causes wet, blackish-brown spots on leaves and stems, caused by fungi or bacteria.</p></div>
            <div class="disease-card"><div class="disease-icon">🌿</div><h4>Soft Rot</h4><p>A very fast and destructive bacterial infection that makes plant tissue soft and foul-smelling.</p></div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("---")
        st.markdown("## 📸 Best Practices for Accurate Detection")
        st.markdown("""
        <div class="tips-grid">
            <div class="tips-card">
                <h4 style="color: #2ecc71; text-align: center; margin-bottom: 1.5rem;">✅ Do This</h4>
                <ul class="dos-donts-list">
                    <li class="dos">🌟 Use bright, natural lighting</li>
                    <li class="dos">🎯 Focus on one specific area</li>
                    <li class="dos">📱 Keep your device steady to avoid blur</li>
                    <li class="dos">🖼️ Use a plain background if possible</li>
                    <li class="dos">📏 Get close enough to show details</li>
                </ul>
            </div>
            <div class="tips-card">
                <h4 style="color: #e74c3c; text-align: center; margin-bottom: 1.5rem;">❌ Avoid This</h4>
                <ul class="dos-donts-list">
                    <li class="donts">🌑 Taking photos in dark or overly bright areas</li>
                    <li class="donts">🏞️ Including too many plant parts in one image</li>
                    <li class="donts">💫 Using blurry or pixelated images</li>
                    <li class="donts">🌫️ Heavy shadows covering the affected areas</li>
                    <li class="donts">📐 Extreme angles that distort plant features</li>
                </ul>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("---")
        st.markdown("""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                     padding: 2rem; border-radius: 20px; text-align: center; margin: 2rem 0;">
            <h3 style="color: white; margin-bottom: 1rem;">🚀 Ready to Get Started?</h3>
            <p style="color: rgba(255,255,255,0.9); margin-bottom: 0;">
                Choose the <strong>Camera</strong> tab to take a live photo, or use the <strong>Upload</strong> tab 
                to analyze an existing image from your device.
            </p>
        </div>
        """, unsafe_allow_html=True)
        st.warning(
            "⚠️ **Important Disclaimer:** This AI tool provides preliminary disease detection and should be used "
            "as a guide. Always consult with a horticultural expert for serious plant health concerns."
        )

    with tab_camera:
        st.markdown("""
        <div class="feature-card">
            <h3>📷 Detect with Camera</h3>
            <p>Take a photo using your camera for instant disease analysis.</p>
        </div>
        """, unsafe_allow_html=True)

        if st.session_state.get('camera_activated', False):
            camera_input = st.camera_input(
                "Point the camera at the orchid plant...", 
                key="camera", 
                label_visibility="collapsed"
            )
            
            if camera_input is not None:
                image = Image.open(camera_input)
                process_and_display_results(image, model)
            
            if st.button("❌ Deactivate Camera"):
                st.session_state.camera_activated = False
                st.rerun()
        else:
            if st.button("📷 Activate Camera"):
                st.session_state.camera_activated = True
                st.rerun()

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
            if st.button("🔍 Analyze Disease", key="upload_analyze"):
                process_and_display_results(image, model)

    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 2rem;">
        <p>🌺 Orchid Disease Detection System | Powered by AI & YOLO</p>
        <p>Developed with ❤️ for orchid enthusiasts</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()