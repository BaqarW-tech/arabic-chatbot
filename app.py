import streamlit as st
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Arabic Q&A Bot", page_icon="🤖")

st.title("🤖 أجب عن سؤالي")
st.markdown("اكتب نصاً عربياً، ثم اسألني سؤالاً عنه!")

# -------------------------
# Text Cleaning
# -------------------------

def clean_arabic(text):
    text = re.sub(r'[^\u0600-\u06FF\s]', '', text)
    return text.strip()

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

    sentences = split_sentences(clean_arabic(context))
    question = clean_arabic(question)

    if len(sentences) == 0:
        return None, 0

    corpus = sentences + [question]

    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(corpus)

    similarity = cosine_similarity(vectors[-1], vectors[:-1])

    best_idx = similarity.argmax()
    best_score = similarity[0][best_idx]

    return sentences[best_idx], float(best_score)

# -------------------------
# UI
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

if st.button("🤖 احصل على الجواب"):

    if context and question:

        with st.spinner("🤔 جاري البحث عن الإجابة..."):

            answer, score = find_answer(context, question)

            if answer:

                st.success("✨ الإجابة:")
                st.markdown(f">>> {answer}")
                st.caption(f"درجة التشابه: {score:.2f}")

            else:
                st.warning("لم يتم العثور على إجابة مناسبة.")

    else:
        st.warning("⚠️ من فضلك اكتب النص والسؤال")