import streamlit as st
import re
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from tashaphyne.stemming import ArabicLightStemmer
import arabicstopwords.arabicstopwords as stopwords
from sentence_transformers import SentenceTransformer

# -------------------------
# Page Configuration
# -------------------------

st.set_page_config(page_title="Arabic Q&A Bot", page_icon="🤖")

st.markdown("""
<style>
body {direction: RTL; text-align: right;}
textarea, input {direction: RTL; text-align: right;}
</style>
""", unsafe_allow_html=True)

st.title("🤖 أجب عن سؤالي")
st.markdown("اكتب نصاً عربياً، ثم اسألني سؤالاً عنه!")

st.info(
"""
مثال:

النص: القاهرة هي عاصمة مصر وتقع على نهر النيل.

السؤال: ما هي عاصمة مصر؟
"""
)

# -------------------------
# Load Embedding Model
# -------------------------

@st.cache_resource
def load_model():
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    return model

model = load_model()

# -------------------------
# Arabic Tools
# -------------------------

stemmer = ArabicLightStemmer()
arabic_stopwords = set(stopwords.stopwords_list())

# -------------------------
# Text Cleaning
# -------------------------

def clean_arabic(text):

    text = re.sub(r'[^\u0600-\u06FF\s]', '', text)
    text = re.sub(r'\s+', ' ', text)

    return text.strip()

# -------------------------
# Stopword Removal
# -------------------------

def remove_stopwords(text):

    words = text.split()
    filtered = [w for w in words if w not in arabic_stopwords]

    return " ".join(filtered)

# -------------------------
# Stemming
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
# Semantic QA Engine
# -------------------------

def find_answer(context, question):

    context = clean_arabic(context)
    question = clean_arabic(question)

    sentences = split_sentences(context)

    if len(sentences) == 0:
        return None, 0

    processed_sentences = []

    for s in sentences:
        s = remove_stopwords(s)
        s = stem_text(s)
        processed_sentences.append(s)

    question = remove_stopwords(question)
    question = stem_text(question)

    sentence_embeddings = model.encode(processed_sentences)
    question_embedding = model.encode([question])

    similarity = cosine_similarity(question_embedding, sentence_embeddings)

    best_index = similarity.argmax()
    best_score = similarity[0][best_index]

    return sentences[best_index], float(best_score)

# -------------------------
# User Input
# -------------------------

context = st.text_area(
    "📄 النص (Context)",
    height=200,
    placeholder="اكتب النص العربي هنا..."
)

question = st.text_input(
    "❓ سؤالك (Question)",
    placeholder="ماذا تريد أن تسأل؟"
)

# -------------------------
# Answer Button
# -------------------------

if st.button("🤖 احصل على الجواب"):

    if context and question:

        with st.spinner("🤔 جاري تحليل السؤال..."):

            answer, score = find_answer(context, question)

            if answer:

                st.success("✨ الإجابة:")
                st.markdown(f">>> {answer}")
                st.caption(f"درجة التشابه: {score:.2f}")

            else:

                st.warning("لم يتم العثور على إجابة مناسبة.")

    else:

        st.warning("⚠️ من فضلك اكتب النص والسؤال")

# -------------------------
# Evaluation Section
# -------------------------

st.markdown("---")
st.subheader("📊 تقييم النموذج")

try:

    df = pd.read_csv("sample_questions.csv")

    if st.button("تشغيل التقييم"):

        correct = 0

        for _, row in df.iterrows():

            predicted, _ = find_answer(row["context"], row["question"])

            if row["expected_answer"] in predicted:
                correct += 1

        accuracy = correct / len(df)

        st.success(f"دقة النظام: {accuracy:.2%}")

except:
    st.caption("لم يتم العثور على ملف sample_questions.csv")