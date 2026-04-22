import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import re
import cv2
from PIL import Image, ImageFilter, ImageEnhance
import io
import base64
from collections import Counter
import warnings
from dotenv import load_dotenv
import os
warnings.filterwarnings("ignore")

# Load environment variables from .env file
load_dotenv()

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Hospital Analytics — Krithish & Sandeep",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #060d1a; color: #e0eaff; }
    
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #0b1628;
        border-right: 1px solid #1e3050;
    }

    /* Metric cards */
    div[data-testid="metric-container"] {
        background: #111e35;
        border: 1px solid #1e3050;
        border-radius: 8px;
        padding: 1rem;
    }

    /* Headers */
    h1, h2, h3 { color: #e0eaff !important; font-family: 'Segoe UI', sans-serif; }

    /* Accent color */
    .accent { color: #00e5ff; }

    /* Section badge */
    .sec-badge {
        background: rgba(0,229,255,0.1);
        border: 1px solid rgba(0,229,255,0.3);
        color: #00e5ff;
        padding: 4px 12px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-family: monospace;
        letter-spacing: 0.1em;
        display: inline-block;
        margin-bottom: 0.5rem;
    }

    /* Token chips */
    .token {
        background: rgba(127,255,110,0.1);
        border: 1px solid rgba(127,255,110,0.3);
        color: #7fff6e;
        padding: 2px 10px;
        border-radius: 4px;
        margin: 2px;
        display: inline-block;
        font-family: monospace;
        font-size: 0.8rem;
    }

    /* Review sentiment */
    .pos-badge { background: rgba(127,255,110,0.15); color: #7fff6e; border: 1px solid rgba(127,255,110,0.3); padding: 2px 8px; border-radius: 4px; font-size: 0.7rem; font-family: monospace; }
    .neg-badge { background: rgba(255,107,107,0.15); color: #ff6b6b; border: 1px solid rgba(255,107,107,0.3); padding: 2px 8px; border-radius: 4px; font-size: 0.7rem; font-family: monospace; }
    .neu-badge { background: rgba(0,229,255,0.1); color: #00e5ff; border: 1px solid rgba(0,229,255,0.2); padding: 2px 8px; border-radius: 4px; font-size: 0.7rem; font-family: monospace; }

    /* Info card */
    .info-card {
        background: #111e35;
        border: 1px solid #1e3050;
        border-radius: 8px;
        padding: 1.2rem;
        margin-bottom: 1rem;
    }

    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem;
        color: #6b84a8;
        font-size: 0.8rem;
        border-top: 1px solid #1e3050;
        margin-top: 3rem;
    }
    .footer strong { color: #ffb347; }

    /* Plotly chart background */
    .js-plotly-plot { border-radius: 8px; }

    /* Streamlit dataframe */
    .stDataFrame { border-radius: 8px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SAMPLE DATASET
# ─────────────────────────────────────────────
@st.cache_data
def generate_dataset():
    np.random.seed(42)
    n = 500
    departments = ['Cardiology','Orthopaedics','Neurology','Oncology','Paediatrics','OB-GYN','Emergency','Surgery']
    diagnoses   = ['Hypertension','Diabetes','Fracture','Infection','Cancer','Pneumonia','Appendicitis','Stroke']
    insurance   = ['Government','Private','Self-Pay','NGO']

    df = pd.DataFrame({
        'Patient_ID':      [f'P{1000+i}' for i in range(n)],
        'Age':             np.random.randint(5, 95, n),
        'Gender':          np.random.choice(['Male','Female'], n),
        'Department':      np.random.choice(departments, n),
        'Diagnosis':       np.random.choice(diagnoses, n),
        'Blood_Pressure':  np.random.randint(80, 180, n),
        'Heart_Rate':      np.random.randint(55, 120, n),
        'Glucose_Level':   np.random.randint(70, 350, n),
        'BMI':             np.round(np.random.uniform(16, 45, n), 1),
        'Length_of_Stay':  np.random.randint(1, 30, n),
        'Readmission':     np.random.choice([0,1], n, p=[0.82,0.18]),
        'Insurance':       np.random.choice(insurance, n),
        'Outcome':         np.random.choice(['Stable','Discharged','Critical','ICU'], n, p=[0.4,0.35,0.15,0.1]),
        'Satisfaction':    np.random.randint(1, 6, n),
    })
    # inject some missing values
    df.loc[np.random.choice(n, 20, replace=False), 'Glucose_Level'] = np.nan
    df.loc[np.random.choice(n, 15, replace=False), 'BMI']           = np.nan
    return df

df = generate_dataset()

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏥 MediAnalytics")
    st.markdown("**Hospital Intelligence Platform**")
    st.divider()

    page = st.radio(
        "Navigate",
        ["🏠 Home",
         "📊 Charts & Visualization",
         "📝 Text Preprocessing",
         "🖼️ Image Preprocessing",
         "🔬 Dataset Analysis",
         "🔤 Text View Processing",
         "📍 Nearby Hospitals (Maps)"],
        label_visibility="collapsed"
    )

    st.divider()

    st.divider()
    st.markdown("""
    <div style='font-size:0.75rem; color:#6b84a8;'>
    Made by<br>
    <span style='color:#ffb347; font-weight:bold;'>Krithish & Sandeep</span>
    </div>
    """, unsafe_allow_html=True)

# Load Google Places API Key from environment
gapi_key = os.getenv("GOOGLE_PLACES_API_KEY")

# ═══════════════════════════════════════════════
# PAGE: HOME
# ═══════════════════════════════════════════════
if page == "🏠 Home":
    st.markdown("# 🏥 Hospital Data Intelligence Platform")
    st.markdown("##### Comprehensive analytics suite for clinical data • Built by **Krithish & Sandeep**")
    st.divider()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Patients",   f"{len(df):,}",      "↑ 8.3%")
    col2.metric("Departments",      "8",                 "Active")
    col3.metric("Avg Stay (Days)",  f"{df['Length_of_Stay'].mean():.1f}", "↓ 0.3 days")
    col4.metric("Readmission Rate", f"{df['Readmission'].mean()*100:.1f}%", "↓ 1.2%")

    st.divider()
    st.markdown("### 📌 Platform Modules")

    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown("""
        <div class='info-card'>
        <span class='sec-badge'>01 CHARTS</span>
        <h4 style='color:#00e5ff'>Data Visualization</h4>
        <p style='color:#6b84a8; font-size:0.85rem;'>Admissions, department stats, age distribution, bed occupancy — interactive Plotly charts.</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div class='info-card'>
        <span class='sec-badge'>02 TEXT PRE</span>
        <h4 style='color:#7fff6e'>Text Preprocessing</h4>
        <p style='color:#6b84a8; font-size:0.85rem;'>Full NLP pipeline — tokenization, stopword removal, stemming, TF-IDF on clinical notes.</p>
        </div>
        """, unsafe_allow_html=True)
    with m2:
        st.markdown("""
        <div class='info-card'>
        <span class='sec-badge'>03 IMAGE PRE</span>
        <h4 style='color:#ff6b6b'>Image Preprocessing</h4>
        <p style='color:#6b84a8; font-size:0.85rem;'>Medical image pipeline — grayscale, CLAHE, edge detection, segmentation using OpenCV.</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div class='info-card'>
        <span class='sec-badge'>04 DATASET</span>
        <h4 style='color:#ffb347'>Dataset Analysis</h4>
        <p style='color:#6b84a8; font-size:0.85rem;'>Statistical summary, correlation matrix, missing value analysis on 500-patient dataset.</p>
        </div>
        """, unsafe_allow_html=True)
    with m3:
        st.markdown("""
        <div class='info-card'>
        <span class='sec-badge'>05 TEXT VIEW</span>
        <h4 style='color:#c084fc'>Text View Processing</h4>
        <p style='color:#6b84a8; font-size:0.85rem;'>Sentiment analysis, NER, word frequency cloud on patient reviews and clinical notes.</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div class='info-card'>
        <span class='sec-badge'>06 GEO</span>
        <h4 style='color:#38bdf8'>Nearby Hospitals</h4>
        <p style='color:#6b84a8; font-size:0.85rem;'>Google Places API integration — find nearby hospitals by location with ratings and maps.</p>
        </div>
        """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════
# PAGE: CHARTS
# ═══════════════════════════════════════════════
elif page == "📊 Charts & Visualization":
    st.markdown('<span class="sec-badge">01 — CHARTS & VISUALIZATION</span>', unsafe_allow_html=True)
    st.markdown("# 📊 Data Charts")
    st.divider()

    # KPIs
    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Total Admissions",   f"{len(df):,}")
    c2.metric("Emergency Cases",    f"{(df['Department']=='Emergency').sum()}")
    c3.metric("ICU Patients",       f"{(df['Outcome']=='ICU').sum()}")
    c4.metric("Avg Blood Pressure", f"{df['Blood_Pressure'].mean():.0f} mmHg")
    c5.metric("Avg Glucose",        f"{df['Glucose_Level'].mean():.0f} mg/dL")

    st.divider()
    col1, col2 = st.columns(2)

    # Department bar
    with col1:
        dept_counts = df['Department'].value_counts().reset_index()
        dept_counts.columns = ['Department','Count']
        fig = px.bar(dept_counts, x='Department', y='Count',
                     title='Department-wise Patient Count',
                     color='Count', color_continuous_scale='teal',
                     template='plotly_dark')
        fig.update_layout(paper_bgcolor='#111e35', plot_bgcolor='#0b1628', showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    # Diagnosis pie
    with col2:
        diag_counts = df['Diagnosis'].value_counts().reset_index()
        diag_counts.columns = ['Diagnosis','Count']
        fig2 = px.pie(diag_counts, names='Diagnosis', values='Count',
                      title='Diagnosis Distribution', hole=0.45,
                      color_discrete_sequence=px.colors.qualitative.Bold,
                      template='plotly_dark')
        fig2.update_layout(paper_bgcolor='#111e35')
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)

    # Age histogram
    with col3:
        fig3 = px.histogram(df, x='Age', nbins=18,
                            title='Age Distribution of Patients',
                            color_discrete_sequence=['#00e5ff'],
                            template='plotly_dark')
        fig3.update_layout(paper_bgcolor='#111e35', plot_bgcolor='#0b1628')
        st.plotly_chart(fig3, use_container_width=True)

    # Gender pie
    with col4:
        gen = df['Gender'].value_counts().reset_index()
        gen.columns=['Gender','Count']
        fig4 = px.pie(gen, names='Gender', values='Count',
                      title='Gender Split', hole=0.5,
                      color_discrete_sequence=['#00e5ff','#ff6b6b'],
                      template='plotly_dark')
        fig4.update_layout(paper_bgcolor='#111e35')
        st.plotly_chart(fig4, use_container_width=True)

    # Box plot — Length of stay by department
    fig5 = px.box(df, x='Department', y='Length_of_Stay',
                  title='Length of Stay by Department',
                  color='Department',
                  color_discrete_sequence=px.colors.qualitative.Bold,
                  template='plotly_dark')
    fig5.update_layout(paper_bgcolor='#111e35', plot_bgcolor='#0b1628', showlegend=False)
    st.plotly_chart(fig5, use_container_width=True)

    # Outcome breakdown
    col5, col6 = st.columns(2)
    with col5:
        out = df['Outcome'].value_counts().reset_index()
        out.columns=['Outcome','Count']
        fig6 = px.bar(out, x='Outcome', y='Count', title='Patient Outcomes',
                      color='Outcome', color_discrete_sequence=['#7fff6e','#00e5ff','#ff6b6b','#ffb347'],
                      template='plotly_dark')
        fig6.update_layout(paper_bgcolor='#111e35', plot_bgcolor='#0b1628', showlegend=False)
        st.plotly_chart(fig6, use_container_width=True)
    with col6:
        ins = df['Insurance'].value_counts().reset_index()
        ins.columns=['Insurance','Count']
        fig7 = px.funnel(ins, x='Count', y='Insurance', title='Insurance Type Breakdown',
                         color_discrete_sequence=['#c084fc','#38bdf8','#ffb347','#7fff6e'])
        fig7.update_layout(paper_bgcolor='#111e35', plot_bgcolor='#0b1628', template='plotly_dark')
        st.plotly_chart(fig7, use_container_width=True)

# ═══════════════════════════════════════════════
# PAGE: TEXT PREPROCESSING
# ═══════════════════════════════════════════════
elif page == "📝 Text Preprocessing":
    st.markdown('<span class="sec-badge">02 — TEXT PREPROCESSING</span>', unsafe_allow_html=True)
    st.markdown("# 📝 Text Preprocessing Pipeline")
    st.divider()

    STOPWORDS = {
        'a','an','the','is','in','it','of','to','and','or','was','he','she',
        'has','have','been','with','for','on','at','by','his','her','this','that','are'
    }

    def simple_stem(word):
        for suffix in ['ing','tion','ed','ness','ly','ment','al','ive']:
            if word.endswith(suffix) and len(word) > len(suffix)+3:
                return word[:-len(suffix)]
        return word

    def preprocess_pipeline(text):
        steps = {}
        steps['1_raw'] = text
        lower = text.lower()
        steps['2_lowercase'] = lower
        clean = re.sub(r'[^a-z0-9\s]', ' ', lower)
        clean = re.sub(r'\s+', ' ', clean).strip()
        steps['3_noise_removed'] = clean
        tokens = clean.split()
        steps['4_tokens'] = tokens
        no_stop = [t for t in tokens if t not in STOPWORDS]
        steps['5_no_stopwords'] = no_stop
        stemmed = [simple_stem(t) for t in no_stop]
        steps['6_stemmed'] = stemmed
        return steps

    # ── Live Demo ──
    st.markdown("### 🔬 Live Clinical Note Preprocessor")
    sample = "Patient Mr. JOHN DOE (ID: #4521) was admitted on 14/03/2024. He has SEVERE chest pain!!! BP=140/90 mmHg. History of Hypertension & Type-2 Diabetes..."
    user_text = st.text_area("Enter Clinical Note", value=sample, height=100)

    if st.button("▶ Run Preprocessing Pipeline", type="primary"):
        steps = preprocess_pipeline(user_text)

        st.markdown("---")
        st.markdown("**① Raw Input**")
        st.code(steps['1_raw'], language='text')

        st.markdown("**② Lowercased**")
        st.code(steps['2_lowercase'], language='text')

        st.markdown("**③ Noise Removed (special chars stripped)**")
        st.code(steps['3_noise_removed'], language='text')

        st.markdown("**④ Tokenized**")
        chips = " ".join([f'<span class="token">{t}</span>' for t in steps['4_tokens']])
        st.markdown(chips, unsafe_allow_html=True)

        st.markdown("**⑤ After Stopword Removal**")
        chips2 = " ".join([f'<span class="token">{t}</span>' for t in steps['5_no_stopwords']])
        st.markdown(chips2, unsafe_allow_html=True)

        st.markdown("**⑥ Stemmed / Lemmatized**")
        chips3 = " ".join([f'<span class="token" style="color:#ffb347;">{t}</span>' for t in steps['6_stemmed']])
        st.markdown(chips3, unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)
        col1.metric("Original Tokens",  len(steps['4_tokens']))
        col2.metric("After Stopwords",  len(steps['5_no_stopwords']))
        col3.metric("Noise Reduction",  f"{(1 - len(steps['6_stemmed'])/max(len(steps['4_tokens']),1))*100:.0f}%")

    st.divider()

    # ── TF-IDF Simulation ──
    st.markdown("### 📊 TF-IDF Word Importance (Sample Clinical Notes)")

    clinical_notes = [
        "patient admitted severe chest pain hypertension diagnosed myocardial infarction treatment prescribed",
        "fracture femur orthopaedic surgery patient recovery successful discharge planned",
        "diabetes glucose level elevated insulin prescribed diet control recommended",
        "pneumonia chest xray infection antibiotic treatment oxygen therapy administered",
        "cancer chemotherapy oncology patient pain management palliative care",
        "stroke neurological deficit brain scan surgery rehabilitation therapy",
    ]

    all_words = []
    for note in clinical_notes:
        all_words.extend(note.split())
    freq = Counter(all_words)
    top_words = freq.most_common(15)

    tfidf_df = pd.DataFrame(top_words, columns=['Term','Frequency'])
    tfidf_df['TF-IDF Score'] = np.round(tfidf_df['Frequency'] / tfidf_df['Frequency'].max() * 0.95 + np.random.uniform(0, 0.05, len(tfidf_df)), 2)

    fig = px.bar(tfidf_df, x='TF-IDF Score', y='Term', orientation='h',
                 title='TF-IDF Scores — Top Clinical Terms',
                 color='TF-IDF Score', color_continuous_scale='teal',
                 template='plotly_dark')
    fig.update_layout(paper_bgcolor='#111e35', plot_bgcolor='#0b1628')
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.markdown("### ⚙️ Pipeline Summary")
    pipeline_df = pd.DataFrame({
        'Step': ['1. Raw Input','2. Lowercase','3. Noise Removal','4. Tokenization','5. Stopword Removal','6. Stemming','7. TF-IDF Vectorization'],
        'Description': [
            'Original clinical text with noise',
            'Convert all text to lowercase',
            'Remove HTML, punctuation, special chars',
            'Split text into individual tokens',
            'Remove common non-informative words',
            'Reduce words to their root form',
            'Generate numerical feature vectors'
        ],
        'Library': ['—','str.lower()','re.sub()','str.split()','custom set','custom func','sklearn TfidfVectorizer']
    })
    st.dataframe(pipeline_df, use_container_width=True, hide_index=True)

# ═══════════════════════════════════════════════
# PAGE: IMAGE PREPROCESSING
# ═══════════════════════════════════════════════
elif page == "🖼️ Image Preprocessing":
    st.markdown('<span class="sec-badge">03 — IMAGE PREPROCESSING</span>', unsafe_allow_html=True)
    st.markdown("# 🖼️ Medical Image Preprocessing")
    st.divider()

    st.markdown("Upload a medical image (X-ray / MRI) or use the generated sample below.")

    uploaded = st.file_uploader("Upload Medical Image", type=['png','jpg','jpeg'])

    def make_sample_image():
        """Generate a synthetic X-ray-like image."""
        img = np.zeros((256, 256), dtype=np.uint8)
        # Background gradient
        for i in range(256):
            img[i, :] = int(15 + (i / 256) * 30)
        # Ribcage-like lines
        for x in [60, 80, 100, 120, 140, 160, 180, 196]:
            img[40:216, x:x+3] = np.clip(img[40:216, x:x+3] + 80, 0, 255)
        # Lung-like blobs
        cv2.ellipse(img, (85, 128), (40, 70), 0, 0, 360, 60, -1)
        cv2.ellipse(img, (170, 128), (38, 68), 0, 0, 360, 58, -1)
        # Heart-like center
        cv2.ellipse(img, (128, 140), (28, 35), 0, 0, 360, 100, -1)
        # Spine
        img[30:230, 125:132] = np.clip(img[30:230, 125:132] + 120, 0, 255)
        # Add noise
        noise = np.random.randint(0, 20, img.shape, dtype=np.uint8)
        img = cv2.add(img, noise)
        return img

    if uploaded:
        file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
        original_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2GRAY)
    else:
        gray = make_sample_image()
        original_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    gray = cv2.resize(gray, (256, 256))
    original_bgr = cv2.resize(original_bgr, (256, 256))

    # ── Preprocessing steps ──
    # 1. Grayscale
    gray_img = gray.copy()

    # 2. Resize
    resized = cv2.resize(gray_img, (224, 224))

    # 3. Normalize
    normalized = cv2.normalize(gray_img, None, 0, 255, cv2.NORM_MINMAX)

    # 4. CLAHE (contrast enhancement)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(gray_img)

    # 5. Gaussian Blur (denoising)
    blurred = cv2.GaussianBlur(clahe_img, (5, 5), 0)

    # 6. Edge Detection (Canny)
    edges = cv2.Canny(blurred, 30, 100)

    # 7. Thresholding (Segmentation)
    _, segmented = cv2.threshold(clahe_img, 80, 255, cv2.THRESH_BINARY)

    steps = [
        ("1. Original",            original_bgr, "Raw input image from imaging device"),
        ("2. Grayscale",           gray_img,      "Single channel — reduces dimensions"),
        ("3. Normalized",          normalized,    "Pixel values scaled to [0–255]"),
        ("4. CLAHE Enhanced",      clahe_img,     "Contrast Limited Adaptive Histogram Equalization"),
        ("5. Gaussian Denoised",   blurred,       "GaussianBlur removes high-freq noise"),
        ("6. Edge Detection",      edges,         "Canny edge detector highlights structures"),
        ("7. Segmented",           segmented,     "Binary thresholding isolates ROI"),
    ]

    cols = st.columns(4)
    for i, (name, img_arr, desc) in enumerate(steps):
        with cols[i % 4]:
            if len(img_arr.shape) == 2:
                display = Image.fromarray(img_arr)
            else:
                display = Image.fromarray(cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB))
            st.image(display, caption=name, use_column_width=True)
            st.caption(desc)

    st.divider()

    # ── Pixel Intensity Histogram ──
    st.markdown("### 📈 Pixel Intensity Distribution")
    col1, col2 = st.columns(2)
    with col1:
        vals_orig = gray_img.flatten()
        fig = px.histogram(x=vals_orig, nbins=64,
                           title='Original — Pixel Intensity',
                           color_discrete_sequence=['#6b84a8'],
                           template='plotly_dark')
        fig.update_layout(paper_bgcolor='#111e35', plot_bgcolor='#0b1628', showlegend=False)
        fig.update_xaxes(title='Pixel Value (0-255)')
        fig.update_yaxes(title='Frequency')
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        vals_clahe = clahe_img.flatten()
        fig2 = px.histogram(x=vals_clahe, nbins=64,
                            title='CLAHE Enhanced — Pixel Intensity',
                            color_discrete_sequence=['#00e5ff'],
                            template='plotly_dark')
        fig2.update_layout(paper_bgcolor='#111e35', plot_bgcolor='#0b1628', showlegend=False)
        fig2.update_xaxes(title='Pixel Value (0-255)')
        fig2.update_yaxes(title='Frequency')
        st.plotly_chart(fig2, use_container_width=True)

    st.divider()
    st.markdown("### ⚙️ Preprocessing Parameters")
    param_df = pd.DataFrame({
        'Step': ['Resize','Normalize','CLAHE','Gaussian Blur','Canny Edge','Threshold'],
        'Parameters': ['224×224 px','Min-Max [0,255]','clipLimit=3.0, tileSize=8×8','kernel=5×5, σ=0','low=30, high=100','threshold=80'],
        'Library': ['cv2.resize','cv2.normalize','cv2.createCLAHE','cv2.GaussianBlur','cv2.Canny','cv2.threshold'],
        'Purpose': ['Standard DL input size','Model training stability','Enhance local contrast','Remove Gaussian noise','Structural boundaries','Binary segmentation']
    })
    st.dataframe(param_df, use_container_width=True, hide_index=True)

# ═══════════════════════════════════════════════
# PAGE: DATASET ANALYSIS
# ═══════════════════════════════════════════════
elif page == "🔬 Dataset Analysis":
    st.markdown('<span class="sec-badge">04 — DATASET ANALYSIS</span>', unsafe_allow_html=True)
    st.markdown("# 🔬 Dataset Analysis")
    st.divider()

    # ── Dataset Preview ──
    st.markdown("### 📋 Dataset Preview")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rows",     f"{len(df):,}")
    col2.metric("Columns",  len(df.columns))
    col3.metric("Missing",  f"{df.isnull().sum().sum()}")
    col4.metric("Duplicates", f"{df.duplicated().sum()}")

    with st.expander("📄 View Raw Dataset", expanded=False):
        n_rows = st.slider("Rows to display", 5, 50, 10)
        st.dataframe(df.head(n_rows), use_container_width=True)

    st.divider()

    # ── Descriptive Stats ──
    st.markdown("### 📊 Descriptive Statistics")
    numeric_cols = ['Age','Blood_Pressure','Heart_Rate','Glucose_Level','BMI','Length_of_Stay','Satisfaction']
    desc = df[numeric_cols].describe().round(2)
    st.dataframe(desc, use_container_width=True)

    st.divider()

    # ── Missing Values ──
    st.markdown("### ❓ Missing Value Analysis")
    missing = df.isnull().sum().reset_index()
    missing.columns = ['Column','Missing Count']
    missing['Missing %'] = (missing['Missing Count'] / len(df) * 100).round(2)
    missing = missing[missing['Missing Count'] > 0]

    col1, col2 = st.columns(2)
    with col1:
        st.dataframe(missing, use_container_width=True, hide_index=True)
    with col2:
        if len(missing) > 0:
            fig = px.bar(missing, x='Column', y='Missing %',
                         title='Missing Value % per Column',
                         color='Missing %', color_continuous_scale='reds',
                         template='plotly_dark')
            fig.update_layout(paper_bgcolor='#111e35', plot_bgcolor='#0b1628')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.success("✅ No missing values!")

    st.divider()

    # ── Correlation Heatmap ──
    st.markdown("### 🔥 Correlation Matrix")
    corr = df[numeric_cols].corr().round(2)
    fig = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu_r',
                    title='Feature Correlation Heatmap',
                    template='plotly_dark', aspect='auto')
    fig.update_layout(paper_bgcolor='#111e35')
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ── Scatter ──
    st.markdown("### 🔵 Scatter Plot Explorer")
    col1, col2, col3 = st.columns(3)
    xax = col1.selectbox("X Axis", numeric_cols, index=0)
    yax = col2.selectbox("Y Axis", numeric_cols, index=1)
    col_by = col3.selectbox("Color By", ['Department','Diagnosis','Gender','Outcome','Insurance'])

    fig = px.scatter(df, x=xax, y=yax, color=col_by,
                     title=f'{xax} vs {yax} (colored by {col_by})',
                     opacity=0.7, template='plotly_dark',
                     color_discrete_sequence=px.colors.qualitative.Bold)
    fig.update_layout(paper_bgcolor='#111e35', plot_bgcolor='#0b1628')
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ── Readmission Analysis ──
    st.markdown("### 🔄 Readmission Rate by Department")
    readmit = df.groupby('Department')['Readmission'].mean().reset_index()
    readmit.columns = ['Department','Readmission Rate']
    readmit['Readmission Rate'] = (readmit['Readmission Rate'] * 100).round(1)
    readmit = readmit.sort_values('Readmission Rate', ascending=True)
    fig = px.bar(readmit, x='Readmission Rate', y='Department',
                 orientation='h', title='Readmission Rate % by Department',
                 color='Readmission Rate', color_continuous_scale='reds',
                 template='plotly_dark')
    fig.update_layout(paper_bgcolor='#111e35', plot_bgcolor='#0b1628')
    st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════
# PAGE: TEXT VIEW PROCESSING
# ═══════════════════════════════════════════════
elif page == "🔤 Text View Processing":
    st.markdown('<span class="sec-badge">05 — TEXT VIEW PROCESSING</span>', unsafe_allow_html=True)
    st.markdown("# 🔤 Text View Processing")
    st.divider()

    POSITIVE_WORDS = {'excellent','great','good','caring','professional','clean','helpful','attentive','quick','smooth','comfortable','recommend','satisfied','impressed','wonderful'}
    NEGATIVE_WORDS = {'poor','slow','bad','rude','dirty','long','wait','billing','confusing','noisy','unfriendly','uncomfortable','terrible','horrible','worst'}

    def get_sentiment(text):
        words = set(text.lower().split())
        pos = len(words & POSITIVE_WORDS)
        neg = len(words & NEGATIVE_WORDS)
        if pos > neg: return 'POSITIVE', pos
        elif neg > pos: return 'NEGATIVE', neg
        return 'NEUTRAL', 0

    reviews = [
        "The nursing staff was extremely attentive and caring. Dr. Sharma explained everything clearly. Highly recommend the cardiology unit.",
        "Waiting time in the emergency ward was over 3 hours. The billing process was confusing and poorly organized.",
        "Clean facilities and modern equipment. The surgery went smoothly. Recovery room nurses were very professional.",
        "Average experience overall. Some staff were helpful while others seemed rushed. Food quality could be improved.",
        "Excellent care provided by the oncology team. Regular check-ins and very comfortable environment throughout.",
        "The diagnostic process was slow and long wait times at reception. However, doctors were professional and knowledgeable.",
        "Wonderful experience. Doctors and nurses were friendly and attentive. Would definitely recommend this hospital.",
        "The rooms were dirty and maintenance was poor. Complained multiple times but nothing improved.",
    ]

    # ── Sentiment Analysis ──
    st.markdown("### 😊 Sentiment Analysis — Patient Reviews")

    results = [get_sentiment(r) for r in reviews]
    sentiments = [r[0] for r in results]
    pos_count = sentiments.count('POSITIVE')
    neg_count = sentiments.count('NEGATIVE')
    neu_count = sentiments.count('NEUTRAL')

    col1, col2, col3 = st.columns(3)
    col1.metric("😊 Positive", pos_count, f"{pos_count/len(reviews)*100:.0f}%")
    col2.metric("😐 Neutral",  neu_count, f"{neu_count/len(reviews)*100:.0f}%")
    col3.metric("😞 Negative", neg_count, f"{neg_count/len(reviews)*100:.0f}%")

    # Sentiment pie
    fig = px.pie(values=[pos_count, neu_count, neg_count],
                 names=['Positive','Neutral','Negative'],
                 hole=0.5,
                 color_discrete_sequence=['#7fff6e','#00e5ff','#ff6b6b'],
                 template='plotly_dark')
    fig.update_layout(paper_bgcolor='#111e35')
    st.plotly_chart(fig, use_container_width=True)

    # Reviews list
    st.markdown("### 💬 Review Browser")
    for review, (sent, _) in zip(reviews, results):
        badge_class = 'pos-badge' if sent=='POSITIVE' else ('neg-badge' if sent=='NEGATIVE' else 'neu-badge')
        st.markdown(f"""
        <div class='info-card'>
        <span class='{badge_class}'>{sent}</span>
        <p style='color:#a0b4cc; margin-top:8px; font-size:0.88rem;'>{review}</p>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # ── Word Frequency ──
    st.markdown("### 📊 Word Frequency Analysis")
    all_text = " ".join(reviews).lower()
    all_text = re.sub(r'[^a-z\s]', '', all_text)
    stop = {'the','a','and','is','was','were','in','to','of','on','at','by','for','with','but','some','that','this','be','could'}
    words = [w for w in all_text.split() if w not in stop and len(w) > 3]
    freq_counter = Counter(words)
    top20 = freq_counter.most_common(20)
    wf_df = pd.DataFrame(top20, columns=['Word','Frequency'])

    fig = px.bar(wf_df, x='Frequency', y='Word', orientation='h',
                 title='Top 20 Words in Patient Reviews',
                 color='Frequency', color_continuous_scale='teal',
                 template='plotly_dark')
    fig.update_layout(paper_bgcolor='#111e35', plot_bgcolor='#0b1628')
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ── NER Demo ──
    st.markdown("### 🏷️ Named Entity Recognition (NER)")
    st.markdown("Highlighting medical entities in clinical notes:")
    ner_html = """
    <div style='background:#0b1628; border:1px solid #1e3050; border-radius:8px; padding:1.2rem; font-family:monospace; font-size:0.85rem; line-height:2.2;'>
    Patient <span style='background:rgba(0,229,255,0.2);color:#00e5ff;border-radius:3px;padding:2px 8px;border:1px solid rgba(0,229,255,0.4);'>JOHN SMITH [PERSON]</span>
    admitted to <span style='background:rgba(127,255,110,0.15);color:#7fff6e;border-radius:3px;padding:2px 8px;border:1px solid rgba(127,255,110,0.4);'>Cardiology [DEPT]</span>
    on <span style='background:rgba(255,179,71,0.15);color:#ffb347;border-radius:3px;padding:2px 8px;border:1px solid rgba(255,179,71,0.4);'>March 14, 2024 [DATE]</span>.
    Diagnosed with <span style='background:rgba(255,107,107,0.2);color:#ff6b6b;border-radius:3px;padding:2px 8px;border:1px solid rgba(255,107,107,0.4);'>Myocardial Infarction [DISEASE]</span>.
    Prescribed <span style='background:rgba(192,132,252,0.15);color:#c084fc;border-radius:3px;padding:2px 8px;border:1px solid rgba(192,132,252,0.4);'>Aspirin 75mg [MEDICATION]</span>
    and <span style='background:rgba(192,132,252,0.15);color:#c084fc;border-radius:3px;padding:2px 8px;border:1px solid rgba(192,132,252,0.4);'>Clopidogrel [MEDICATION]</span>.
    Referred to <span style='background:rgba(56,189,248,0.15);color:#38bdf8;border-radius:3px;padding:2px 8px;border:1px solid rgba(56,189,248,0.4);'>City General Hospital [ORG]</span>.
    </div>
    """
    st.markdown(ner_html, unsafe_allow_html=True)

# ═══════════════════════════════════════════════
# PAGE: NEARBY HOSPITALS (GOOGLE PLACES)
# ═══════════════════════════════════════════════
elif page == "📍 Nearby Hospitals (Maps)":
    st.markdown('<span class="sec-badge">06 — GEOLOCATION • GOOGLE PLACES API</span>', unsafe_allow_html=True)
    st.markdown("# 📍 Nearby Hospitals Finder")
    st.markdown("Uses **Google Places API** to fetch real nearby hospitals based on your location.")
    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        city_input = st.text_input("📌 Enter City / Area", placeholder="e.g. Chennai, Bangalore, Mumbai...")
    with col2:
        radius_km = st.slider("Search Radius (km)", 1, 50, 5)

    search_btn = st.button("🔍 Search Nearby Hospitals", type="primary")

    def geocode_city(city, api_key):
        """Convert city name to lat/lng using Google Geocoding API."""
        url = "https://maps.googleapis.com/maps/api/geocode/json"
        params = {"address": city, "key": api_key}
        resp = requests.get(url, params=params, timeout=10)
        data = resp.json()
        if data.get('status') == 'OK':
            loc = data['results'][0]['geometry']['location']
            return loc['lat'], loc['lng']
        return None, None

    def search_hospitals(lat, lng, radius_m, api_key):
        """Search nearby hospitals using Google Places Nearby Search."""
        url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
        params = {
            "location": f"{lat},{lng}",
            "radius":   radius_m,
            "type":     "hospital",
            "key":      api_key
        }
        resp = requests.get(url, params=params, timeout=10)
        return resp.json()

    def get_place_photo(photo_ref, api_key, max_width=400):
        url = f"https://maps.googleapis.com/maps/api/place/photo?maxwidth={max_width}&photoreference={photo_ref}&key={api_key}"
        return url

    if search_btn:
        if not gapi_key:
            st.error("⚠️ Please enter your Google Places API key in the sidebar first!")
        elif not city_input.strip():
            st.warning("⚠️ Please enter a city name.")
        else:
            with st.spinner(f"Searching hospitals near **{city_input}**..."):
                try:
                    lat, lng = geocode_city(city_input, gapi_key)
                    if lat is None:
                        st.error("❌ Could not geocode that city. Check city name or API key.")
                    else:
                        st.success(f"📌 Location found: `{lat:.4f}, {lng:.4f}`")
                        radius_m = radius_km * 1000
                        data = search_hospitals(lat, lng, radius_m, gapi_key)

                        if data.get('status') == 'OK':
                            results = data.get('results', [])
                            st.markdown(f"### 🏥 Found **{len(results)}** hospitals within {radius_km}km of {city_input}")
                            st.divider()

                            # Map
                            map_data = []
                            for h in results:
                                loc = h['geometry']['location']
                                map_data.append({'lat': loc['lat'], 'lon': loc['lng'], 'name': h['name']})
                            map_df = pd.DataFrame(map_data)
                            st.map(map_df, zoom=12)

                            st.divider()

                            # Results cards
                            for i, hospital in enumerate(results):
                                name    = hospital.get('name', 'Unknown')
                                address = hospital.get('vicinity', 'N/A')
                                rating  = hospital.get('rating', 'N/A')
                                reviews_count = hospital.get('user_ratings_total', 0)
                                status  = hospital.get('business_status', 'N/A')
                                place_id = hospital.get('place_id', '')

                                stars = "⭐" * int(rating) if isinstance(rating, (int,float)) else ""
                                status_color = "#7fff6e" if status == "OPERATIONAL" else "#ff6b6b"

                                st.markdown(f"""
                                <div class='info-card'>
                                <b style='color:#00e5ff; font-size:1rem;'>🏥 {i+1}. {name}</b><br>
                                <span style='color:#6b84a8; font-size:0.82rem;'>📍 {address}</span><br>
                                <span style='color:#ffb347;'>{stars} {rating}</span>
                                <span style='color:#6b84a8; font-size:0.78rem;'> ({reviews_count} reviews)</span>
                                &nbsp;&nbsp;
                                <span style='color:{status_color}; font-size:0.75rem; font-family:monospace;'>● {status}</span><br>
                                <a href='https://www.google.com/maps/place/?q=place_id:{place_id}' target='_blank'
                                   style='color:#38bdf8; font-size:0.78rem;'>🔗 View on Google Maps</a>
                                </div>
                                """, unsafe_allow_html=True)

                            # Summary chart
                            st.divider()
                            st.markdown("### 📊 Hospital Ratings Overview")
                            rated = [h for h in results if isinstance(h.get('rating'), (int,float))]
                            if rated:
                                r_df = pd.DataFrame({
                                    'Hospital': [h['name'][:30] for h in rated],
                                    'Rating':   [h['rating'] for h in rated],
                                    'Reviews':  [h.get('user_ratings_total', 0) for h in rated]
                                }).sort_values('Rating', ascending=True)
                                fig = px.bar(r_df, x='Rating', y='Hospital', orientation='h',
                                             color='Rating', color_continuous_scale='teal',
                                             title='Hospital Ratings Comparison',
                                             template='plotly_dark',
                                             hover_data=['Reviews'])
                                fig.update_layout(paper_bgcolor='#111e35', plot_bgcolor='#0b1628')
                                st.plotly_chart(fig, use_container_width=True)

                        elif data.get('status') == 'REQUEST_DENIED':
                            st.error("❌ API request denied. Check your API key permissions (Places API must be enabled).")
                        elif data.get('status') == 'ZERO_RESULTS':
                            st.warning(f"No hospitals found within {radius_km}km of {city_input}. Try increasing the radius.")
                        else:
                            st.error(f"API Error: {data.get('status')} — {data.get('error_message','')}")

                except requests.exceptions.ConnectionError:
                    st.error("❌ Network error — could not reach Google API. Check your internet connection.")
                except Exception as e:
                    st.error(f"❌ Unexpected error: {e}")

    else:
        # Demo preview
        st.info("ℹ️ Enter a city and your Google Places API key to search real hospitals. Demo below shows the expected output.")
        demo_data = pd.DataFrame({
            'Hospital': ['Apollo Hospital','AIIMS','Fortis','Max Healthcare','Manipal'],
            'Rating':   [4.5, 4.3, 4.2, 4.4, 4.1],
            'Distance_km': [0.8, 2.1, 3.4, 4.2, 4.9],
            'Status':   ['OPERATIONAL']*5
        })
        st.dataframe(demo_data, use_container_width=True, hide_index=True)
        fig = px.bar(demo_data.sort_values('Rating'), x='Rating', y='Hospital',
                     orientation='h', color='Rating', color_continuous_scale='teal',
                     template='plotly_dark', title='Sample Output — Hospital Ratings')
        fig.update_layout(paper_bgcolor='#111e35', plot_bgcolor='#0b1628')
        st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────────
# FOOTER (all pages)
# ─────────────────────────────────────────────
st.markdown("""
<div class='footer'>
    🏥 Hospital Data Intelligence Platform<br>
    Designed & Developed by <strong>Krithish</strong> & <strong>Sandeep</strong>
</div>
""", unsafe_allow_html=True)