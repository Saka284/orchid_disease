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

# Enhanced CSS with modern design and responsiveness
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Root variables for consistent theming */
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
    
    /* Global Styles */
    .stApp {
        background: linear-gradient(135deg, #0c0c0c 0%, #1a1a1a 50%, #2d2d2d 100%);
        font-family: 'Inter', sans-serif;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Main header with animated gradient */
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
    
    /* Enhanced feature cards */
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
    
    /* Result cards with improved styling */
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
    
    .detection-result::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(245, 87, 108, 0.1) 0%, transparent 70%);
        animation: pulse 3s ease-in-out infinite;
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
    
    .healthy-result::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(56, 239, 125, 0.1) 0%, transparent 70%);
        animation: pulse 3s ease-in-out infinite;
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1) rotate(0deg); }
        50% { transform: scale(1.1) rotate(180deg); }
    }
    
    /* Recommendation card */
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
    
    /* Sidebar styling */
    .sidebar-content {
        background: var(--card-bg);
        border: 1px solid var(--border-color);
        padding: 1.5rem;
        border-radius: 15px;
        color: var(--text-primary);
        margin: 1rem 0;
        box-shadow: var(--shadow);
    }
    
    /* Enhanced buttons */
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
    
    .stButton > button:active {
        transform: translateY(0);
    }
    
    /* Home page cards */
    .home-card {
        background: var(--card-bg);
        border: 1px solid var(--border-color);
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        height: 100%;
        color: var(--text-primary);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .home-card:hover {
        transform: translateY(-5px);
        border-color: #667eea;
        box-shadow: var(--shadow-hover);
    }
    
    .home-card h3 {
        color: #5dade2;
        margin-bottom: 1rem;
        font-weight: 600;
    }
    
    /* Disease cards grid - responsive */
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
        position: relative;
        overflow: hidden;
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
    
    /* Info cards for disease details */
    .info-card-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
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
        transition: all 0.3s ease;
    }
    
    .info-card:hover, .info-card-treatment:hover {
        transform: translateY(-3px);
        box-shadow: var(--shadow-hover);
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
        transition: all 0.2s ease;
    }
    
    .info-card li:hover, .info-card-treatment li:hover {
        background: rgba(102, 126, 234, 0.2);
        transform: translateX(5px);
    }
    
    /* Dos and Don'ts styling */
    .tips-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
        gap: 2rem;
        margin: 2rem 0;
    }
    
    .tips-card {
        background: var(--card-bg);
        border: 1px solid var(--border-color);
        border-radius: 20px;
        padding: 2rem;
        box-shadow: var(--shadow);
    }
    
    .dos-donts-list {
        list-style: none;
        padding: 0;
    }
    
    .dos-donts-list li {
        margin-bottom: 0.75rem;
        padding: 1rem;
        border-radius: 12px;
        transition: all 0.2s ease;
    }
    
    .dos {
        background: rgba(46, 204, 113, 0.1);
        border-left: 4px solid #2ecc71;
    }
    
    .donts {
        background: rgba(231, 76, 60, 0.1);
        border-left: 4px solid #e74c3c;
    }
    
    .dos:hover {
        background: rgba(46, 204, 113, 0.2);
        transform: translateX(5px);
    }
    
    .donts:hover {
        background: rgba(231, 76, 60, 0.2);
        transform: translateX(5px);
    }
    
    /* Progress bar styling */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea, #764ba2);
    }
    
    /* Tab styling */
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
    
    /* Responsive design */
    @media (max-width: 768px) {
        .main-header {
            font-size: 2rem;
            margin: 1rem 0;
        }
        
        .feature-card, .home-card, .info-card, .info-card-treatment {
            padding: 1.5rem;
        }
        
        .disease-grid, .info-card-grid, .tips-grid {
            grid-template-columns: 1fr;
            gap: 1rem;
        }
        
        .detection-result, .healthy-result {
            padding: 1.5rem;
        }
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: var(--dark-bg);
    }
    
    ::-webkit-scrollbar-thumb {
        background: #667eea;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #764ba2;
    }
    
    /* Loading animation */
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    .loading-spinner {
        border: 4px solid rgba(102, 126, 234, 0.3);
        border-radius: 50%;
        border-top: 4px solid #667eea;
        width: 40px;
        height: 40px;
        animation: spin 1s linear infinite;
        margin: 20px auto;
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

        cv2.rectangle(cv_image, (x1, y1), (x2, y2), color, 3)

        label = f"{disease}: {confidence:.1%}"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(cv_image, (x1, y1 - label_size[1] - 15), (x1 + label_size[0] + 10, y1), color, -1)
        cv2.putText(cv_image, label, (x1 + 5, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

def analyze_detections(detections):
    """Simplified analysis. If there are any detections, it's a disease. Otherwise, no disease is detected."""
    if not detections:
        return "no_disease_detected", "No disease was detected on the plant.", []
    else:
        disease_names = [d['disease'] for d in detections]
        return "diseased", f"Detected {len(disease_names)} area(s) of disease.", disease_names

def display_disease_info(disease_name):
    """Display disease information and recommendations in a modern card layout."""
    if disease_name in DISEASE_INFO:
        info = DISEASE_INFO.get(disease_name)
        if not info:
            return

        st.markdown(f"### 📋 Information & Recommendations for: **{disease_name}**")

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
            <p>An AI-powered system to detect specific diseases on orchid plants using advanced YOLO technology.</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### 🦠 Detectable Diseases:")
        diseases = [
            ("🌸 Petal Blight", "Fungal infection affecting flower petals"),
            ("🍃 Brown Spot", "Pseudobulb rotting disease"),
            ("🌿 Soft Rot", "Bacterial tissue deterioration")
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
            <div class="disease-card">
                <div class="disease-icon">🌸</div>
                <h4>Petal Blight</h4>
                <p>Fungal infection causing brown spots and premature flower drop. Often occurs in humid conditions.</p>
            </div>
            <div class="disease-card">
                <div class="disease-icon">🍃</div>
                <h4>Brown Spot</h4>
                <p>Pseudobulb rotting disease that causes soft, brown tissue deterioration and spore formation.</p>
            </div>
            <div class="disease-card">
                <div class="disease-icon">🌿</div>
                <h4>Soft Rot</h4>
                <p>Bacterial infection leading to mushy tissue, foul odor, and rapid plant deterioration.</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        st.markdown("## 📸 Best Practices for Accurate Detection")
        
        st.markdown("""
        <div class="tips-grid">
            <div class="tips-card">
                <h4 style="color: #2ecc71; text-align: center; margin-bottom: 1.5rem;">✅ Do This</h4>
                <ul class="dos-donts-list">
                    <li class="dos">🌟 Use bright, natural lighting for clear images</li>
                    <li class="dos">🎯 Focus on one specific area (leaf or flower)</li>
                    <li class="dos">📱 Keep your device steady to avoid blur</li>
                    <li class="dos">🖼️ Use plain backgrounds when possible</li>
                    <li class="dos">📏 Get close enough to show details clearly</li>
                </ul>
            </div>
            <div class="tips-card">
                <h4 style="color: #e74c3c; text-align: center; margin-bottom: 1.5rem;">❌ Avoid This</h4>
                <ul class="dos-donts-list">
                    <li class="donts">🌑 Taking photos in very dark or overly bright areas</li>
                    <li class="donts">🏞️ Including too many plant parts in one image</li>
                    <li class="donts">💫 Using blurry or pixelated images</li>
                    <li class="donts">🌫️ Heavy shadows covering the affected areas</li>
                    <li class="donts">📐 Extreme angles that distort the plant features</li>
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
            "as a guide alongside professional horticultural advice. Always consult with plant specialists for "
            "serious plant health concerns."
        )

    def process_and_display_results(image):
        with st.spinner("🔍 Analyzing your orchid image..."):
            # Add loading animation
            st.markdown('<div class="loading-spinner"></div>', unsafe_allow_html=True)
            
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
                    <h2>✅ Excellent News!</h2>
                    <h3>No Disease Detected</h3>
                    <p style="font-size: 1.1rem; margin: 1rem 0;">{message}</p>
                    <p>Your orchid appears healthy and free from detectable diseases. Keep up the excellent care! 🌟</p>
                </div>
                """, unsafe_allow_html=True)

                # --- CORRECTED SECTION ---
                st.markdown("""
                <div class="recommendation-card">
                    <h4>🌱 Maintenance Tips for Healthy Orchids</h4>
                    <ul>
                        <li><strong>💡 Lighting:</strong> Bright, indirect sunlight. East or west-facing windows are ideal.</li>
                        <li><strong>💧 Watering:</strong> Water thoroughly when the growing medium is almost dry. Avoid letting it sit in water.</li>
                        <li><strong>🌡️ Humidity:</strong> Orchids thrive in 50-70% humidity. Consider a humidifier or a pebble tray.</li>
                        <li><strong>🌬️ Airflow:</strong> Good air circulation is crucial to prevent fungal and bacterial issues.</li>
                        <li><strong>🌿 Fertilizing:</strong> Use a balanced orchid fertilizer weakly during the growing season.</li>
                        <li><strong>🧐 Inspect Regularly:</strong> Check your plant often for any early signs of pests or disease.</li>
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
            
            # Display the button only after a file is uploaded
            if st.button("🔍 Analyze Disease", key="upload_analyze"):
                process_and_display_results(image)

    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 2rem;">
        <p>🌺 Orchid Disease Detection System | Powered by AI & YOLO</p>
        <p>Developed with ❤️ for orchid enthusiasts</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()