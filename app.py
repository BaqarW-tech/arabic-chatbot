import streamlit as st
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tashaphyne.stemming import ArabicLightStemmer

# -------------------------
# Page Config
# -------------------------

st.set_page_config(page_title="Arabic Q&A Bot", page_icon="🤖")

# -------------------------
# RTL Layout
# -------------------------

st.markdown("""
<style>
body {
direction: RTL;
text-align: right;
}
textarea, input {
direction: RTL;
text-align: right;
}
</style>
""", unsafe_allow_html=True)

# -------------------------
# Title
# -------------------------

st.title("🤖 أجب عن سؤالي")
st.markdown("اكتب نصاً عربياً، ثم اسألني سؤالاً عنه!")

st.info(
"""
مثال للاستخدام:

النص: القاهرة هي عاصمة مصر وتقع على نهر النيل.

السؤال: ما هي عاصمة مصر؟
"""
)

# -------------------------
# Arabic Stemmer
# -------------------------

stemmer = ArabicLightStemmer()

# -------------------------
# Cleaning Arabic Text
# -------------------------

def clean_arabic(text):

    text = re.sub(r'[^\u0600-\u06FF\s]', '', text)
    text = re.sub(r'\s+', ' ', text)

    return text.strip()

# -------------------------
# Arabic Stemming
# -------------------------

def stem_text(text):

    words = text.split()
    stems = []

    for w in words:
        stemmer.light_stem(w)
        stems.append(stemmer.get_stem())

    return " ".join(stems)

# -------------------------
# Sentence Split
# -------------------------

def split_sentences(text):

    sentences = re.split(r'[.!؟\n]', text)
    return [s.strip() for s in sentences if s.strip()]

# -------------------------
# QA Engine
# -------------------------

def find_answer(context, question):

    context = clean_arabic(context)
    question = clean_ar