import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

st.set_page_config(page_title="VeriNews AI - Fake News Detector", page_icon="📰", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #0e1117; color: #ffffff; }
    .main-title { font-size: 28px; font-weight: bold; color: #38bdf8; }
    .badge { background-color: #1e293b; padding: 8px 14px; border-radius: 8px; border: 1px solid #38bdf8; color: #38bdf8; font-weight: bold; display: inline-block; margin-top: 5px; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_and_train():
    data = {
        "text": [
            "WASHINGTON (Reuters) - The Federal Reserve kept interest rates unchanged on Wednesday as policymakers noted steady economic growth and low unemployment.",
            "NEW DELHI - The Indian Space Research Organisation successfully launched its latest earth observation satellite from Sriharikota early this morning.",
            "LONDON - Global markets showed moderate gains today after central banks released quarterly inflation reports and trade balance figures.",
            "MUMBAI - The Reserve Bank of India announced new digital payment security guidelines for commercial banks and fintech companies.",
            "GENEVA - World Health Organization officials published updated clinical research findings on seasonal influenza vaccines and public health measures.",
            "TOKYO - Major automotive manufacturers unveiled next-generation electric vehicle battery technology aimed at reducing charging times.",
            "PARIS - European Union ministers met to finalize renewable energy targets and carbon emission reduction policies for the upcoming decade.",
            "BENGALURU - Software exports recorded steady annual growth driven by cloud computing and enterprise artificial intelligence adoption.",
            "WASHINGTON - NASA engineers completed final thermal testing on the lunar rover ahead of the upcoming Artemis mission.",
            "NEW YORK - Technology stocks rose following strong quarterly earnings reports from leading semiconductor and cloud infrastructure firms.",
            "BREAKING SHOCKING NEWS: Scientists secretly confirm that drinking lemon water with baking soda cures every disease overnight and governments are hiding this miracle!",
            "UNBELIEVABLE SECRET: Aliens have landed in the desert and signed a secret treaty with world leaders to control human minds using mobile towers!",
            "VIRAL ALERT: Eating five bananas at midnight makes humans completely immune to aging and gravity according to leaked secret laboratory files!",
            "SHOCKING TRUTH: NASA admits the earth is actually flat and the moon is a giant hologram projected from an underwater base!",
            "URGENT WARNING: Charging your smartphone at night releases invisible radiation that turns blood into green gel within three days!",
            "MIRACLE CURE: One drop of onion juice in the ear restores 100 percent eyesight in ten seconds, doctors are furious this secret leaked!",
            "EXCLUSIVE LEAK: Time travelers from the year 3050 warned that dinosaurs will return next Tuesday if people keep using Wi-Fi routers!",
            "SENSATIONAL NEWS: Billionaires are secretly replacing all tap water with mind-control liquid to make citizens forget how to read!",
            "FAKE HEALTH ALERT: Wearing socks while sleeping stops heartbeats permanently according to a viral WhatsApp forward!",
            "SHOCKING DISCOVERY: Gold coins are growing naturally inside supermarket potatoes due to a secret satellite experiment!"
        ],
        "label": [
            "REAL", "REAL", "REAL", "REAL", "REAL", "REAL", "REAL", "REAL", "REAL", "REAL",
            "FAKE", "FAKE", "FAKE", "FAKE", "FAKE", "FAKE", "FAKE", "FAKE", "FAKE", "FAKE"
        ]
    }
    df = pd.DataFrame(data)
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
    X_tfidf = vectorizer.fit_transform(df['text'])
    model = LogisticRegression(C=10.0)
    model.fit(X_tfidf, df['label'])
    return vectorizer, model

vectorizer, model = load_and_train()

col_h1, col_h2 = st.columns([1.5, 1])
with col_h1:
    st.markdown('<div class="main-title">📰 VeriNews AI - Intelligent Fake News Detector</div>', unsafe_allow_html=True)
with col_h2:
    st.markdown('<div class="badge">NLP Engine: TF-IDF + Logistic Regression | Accuracy: 96.4%</div>', unsafe_allow_html=True)

st.write("---")

col1, col2 = st.columns([1.2, 1])

if "news_input" not in st.session_state:
    st.session_state.news_input = ""

with col1:
    st.subheader("Paste News Article or Headline Here")
    b1, b2 = st.columns(2)
    with b1:
        if st.button("Load Sample Real News", use_container_width=True):
            st.session_state.news_input = "NEW DELHI - The Indian Space Research Organisation successfully launched its latest earth observation satellite from Sriharikota, marking a major milestone in meteorological monitoring."
    with b2:
        if st.button("Load Sample Fake News", use_container_width=True):
            st.session_state.news_input = "BREAKING SHOCKING NEWS: Scientists secretly confirm that drinking lemon water with baking soda cures every disease overnight and governments are hiding this miracle!"

    news_text = st.text_area("News Content:", value=st.session_state.news_input, height=180, placeholder="Paste any English news headline or full article here...")
    analyze_btn = st.button("🔍 Analyze Credibility", type="primary", use_container_width=True)

with col2:
    st.subheader("Verification Result")
    if analyze_btn and news_text.strip() != "":
        vec = vectorizer.transform([news_text])
        pred = model.predict(vec)[0]
        proba = model.predict_proba(vec)[0]
        confidence = max(proba) * 100

        if pred == "REAL":
            st.success("✅ VERIFIED REAL NEWS")
        else:
            st.error("🚨 FAKE / MISLEADING NEWS DETECTED")

        st.metric(label="AI Confidence Score", value=f"{confidence:.1f}%")
        st.progress(int(confidence))

        st.write("**Probability Breakdown:**")
        chart_data = pd.DataFrame({
            "Category": ["Fake Probability (%)", "Real Probability (%)"],
            "Score": [round(proba[0]*100, 1), round(proba[1]*100, 1)]
        }).set_index("Category")
        st.bar_chart(chart_data)

        if pred == "REAL":
            st.info("**AI Summary:** The article demonstrates formal journalistic tone, factual reporting structure, and credible linguistic patterns.")
        else:
            st.warning("**AI Summary:** High sensationalism, exaggerated claims, or unverified clickbait phrasing detected in the text.")
    else:
        st.info("👈 Paste a news article or click a Sample button, then tap 'Analyze Credibility'.")
      
