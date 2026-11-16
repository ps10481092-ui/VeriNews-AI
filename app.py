import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

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
    url = "https://githubusercontent.com"
    df = pd.read_csv(url)
    df['content'] = df['title'].fillna('') + " " + df['text'].fillna('')
    X_train, X_test, y_train, y_test = train_test_split(df['content'], df['label'], test_size=0.2, random_state=42)
    vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_tfidf, y_train)
    acc = accuracy_score(y_test, model.predict(X_test_tfidf))
    return vectorizer, model, acc

with st.spinner("Initializing VeriNews NLP Engine..."):
    vectorizer, model, acc = load_and_train()

col_h1, col_h2 = st.columns([1.5, 1])
with col_h1:
    st.markdown('<div class="main-title">📰 VeriNews AI - Intelligent Fake News Detector</div>', unsafe_allow_html=True)
with col_h2:
    st.markdown(f'<div class="badge">NLP Engine: TF-IDF + Logistic Regression | Accuracy: {acc*100:.1f}%</div>', unsafe_allow_html=True)

st.write("---")

col1, col2 = st.columns([1.2, 1])

if "news_input" not in st.session_state:
    st.session_state.news_input = ""

with col1:
    st.subheader("Paste News Article or Headline Here")
    b1, b2 = st.columns(2)
    with b1:
        if st.button("Load Sample Real News", use_container_width=True):
            st.session_state.news_input = "WASHINGTON (Reuters) - The Federal Reserve kept interest rates unchanged on Wednesday as policymakers noted steady economic growth and low unemployment across the United States."
    with b2:
        if st.button("Load Sample Fake News", use_container_width=True):
            st.session_state.news_input = "BREAKING SHOCKING NEWS: Scientists secretly confirm that drinking lemon water with baking soda cures every disease overnight and governments are hiding this miracle!"

    news_text = st.text_area("News Content:", value=st.session_state.news_input, height=200, placeholder="Paste any English news headline or full article here...")
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
            st.info("**AI Summary:** The article shows balanced journalistic tone, factual reporting style, and consistent linguistic patterns.")
        else:
            st.warning("**AI Summary:** High sensationalism, exaggerated claims, or unverified clickbait phrasing detected in the text.")
    else:
        st.info("👈 Paste a news article or click a Sample button, then tap 'Analyze Credibility'.")
      
