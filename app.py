import streamlit as st
import re
import math
from collections import Counter
import csv
import io
from datetime import datetime

# ─────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────
st.set_page_config(
    page_title="بوت الأسئلة والأجوبة العربي",
    page_icon="🤖",
    layout="wide"
)

# ─────────────────────────────────────────
# RTL + Arabic Styling
# ─────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Tajawal:wght@400;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Tajawal', sans-serif;
    }
    .rtl-text {
        direction: rtl;
        text-align: right;
        font-family: 'Tajawal', sans-serif;
        font-size: 16px;
        line-height: 1.8;
    }
    .answer-card {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border: 1px solid #0f3460;
        border-radius: 12px;
        padding: 16px 20px;
        margin: 8px 0;
        direction: rtl;
        text-align: right;
        font-family: 'Tajawal', sans-serif;
    }
    .confidence-bar {
        height: 6px;
        border-radius: 3px;
        background: linear-gradient(90deg, #e94560, #0f3460);
        margin-top: 8px;
    }
    .chat-bubble-user {
        background: #0f3460;
        border-radius: 18px 18px 4px 18px;
        padding: 10px 16px;
        margin: 6px 0;
        direction: rtl;
        text-align: right;
        font-family: 'Tajawal', sans-serif;
        color: white;
    }
    .chat-bubble-bot {
        background: #1a1a2e;
        border: 1px solid #0f3460;
        border-radius: 18px 18px 18px 4px;
        padding: 10px 16px;
        margin: 6px 0;
        direction: rtl;
        text-align: right;
        font-family: 'Tajawal', sans-serif;
        color: #e0e0e0;
    }
    .header-title {
        font-family: 'Tajawal', sans-serif;
        font-size: 2.2rem;
        font-weight: 700;
        color: #e94560;
        text-align: center;
    }
    .subtitle {
        font-family: 'Tajawal', sans-serif;
        text-align: center;
        color: #aaa;
        margin-bottom: 20px;
    }
    .stTextArea textarea {
        direction: rtl;
        font-family: 'Tajawal', sans-serif;
        font-size: 15px;
    }
    .stTextInput input {
        direction: rtl;
        font-family: 'Tajawal', sans-serif;
        font-size: 15px;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# Built-in Arabic Stopwords (no external pkg)
# ─────────────────────────────────────────
ARABIC_STOPWORDS = set([
    "من", "إلى", "عن", "على", "في", "مع", "هذا", "هذه", "ذلك", "تلك",
    "هو", "هي", "هم", "هن", "أنا", "نحن", "أنت", "أنتم", "كان", "كانت",
    "يكون", "تكون", "قد", "لقد", "إن", "أن", "لأن", "حتى", "لكن", "بل",
    "أو", "و", "ف", "ثم", "كل", "بعض", "جميع", "غير", "عند", "لدى",
    "بين", "فوق", "تحت", "قبل", "بعد", "خلال", "حول", "لم", "لن", "لا",
    "ما", "ليس", "كما", "إذا", "حيث", "التي", "الذي", "الذين", "اللواتي",
    "الآن", "هنا", "هناك", "كيف", "متى", "أين", "لماذا", "ماذا", "من",
    "ال", "وال", "بال", "كال", "فال", "لل"
])

# ─────────────────────────────────────────
# NLP Utilities
# ─────────────────────────────────────────
def clean_arabic(text):
    """Normalize Arabic text."""
    text = re.sub(r'[إأآا]', 'ا', text)
    text = re.sub(r'ة', 'ه', text)
    text = re.sub(r'ى', 'ي', text)
    text = re.sub(r'ؤ', 'و', text)
    text = re.sub(r'ئ', 'ي', text)
    text = re.sub(r'[\u064B-\u065F]', '', text)  # remove diacritics
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def tokenize(text):
    """Tokenize and remove stopwords."""
    words = clean_arabic(text).split()
    return [w for w in words if w not in ARABIC_STOPWORDS and len(w) > 1]

def split_sentences(text):
    """Split Arabic text into sentences."""
    sentences = re.split(r'[.!?؟،\n]+', text)
    return [s.strip() for s in sentences if len(s.strip()) > 5]

def compute_tfidf(sentences):
    """Compute TF-IDF vectors for a list of sentences."""
    tokenized = [tokenize(s) for s in sentences]
    # IDF
    doc_count = len(tokenized)
    df = Counter()
    for tokens in tokenized:
        df.update(set(tokens))
    idf = {word: math.log((doc_count + 1) / (freq + 1)) + 1
           for word, freq in df.items()}
    # TF-IDF vectors
    vectors = []
    for tokens in tokenized:
        tf = Counter(tokens)
        total = len(tokens) if tokens else 1
        vec = {w: (count / total) * idf.get(w, 1) for w, count in tf.items()}
        vectors.append(vec)
    return vectors, idf

def cosine_similarity(vec1, vec2):
    """Cosine similarity between two TF-IDF dicts."""
    if not vec1 or not vec2:
        return 0.0
    common = set(vec1.keys()) & set(vec2.keys())
    dot = sum(vec1[w] * vec2[w] for w in common)
    norm1 = math.sqrt(sum(v ** 2 for v in vec1.values()))
    norm2 = math.sqrt(sum(v ** 2 for v in vec2.values()))
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)

def get_top_answers(question, sentences, vectors, idf, top_n=3):
    """Return top N answers with confidence scores."""
    q_tokens = tokenize(question)
    total = len(q_tokens) if q_tokens else 1
    tf_q = Counter(q_tokens)
    q_vec = {w: (count / total) * idf.get(w, 1) for w, count in tf_q.items()}

    scored = []
    for i, (sent, vec) in enumerate(zip(sentences, vectors)):
        score = cosine_similarity(q_vec, vec)
        scored.append((score, i, sent))

    scored.sort(reverse=True)
    results = []
    seen = set()
    for score, idx, sent in scored:
        if sent not in seen and score > 0:
            results.append((sent, round(score * 100, 1)))
            seen.add(sent)
        if len(results) >= top_n:
            break
    return results

# ─────────────────────────────────────────
# Session State
# ─────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "context_text" not in st.session_state:
    st.session_state.context_text = ""
if "sentences" not in st.session_state:
    st.session_state.sentences = []
if "vectors" not in st.session_state:
    st.session_state.vectors = []
if "idf" not in st.session_state:
    st.session_state.idf = {}

# ─────────────────────────────────────────
# Header
# ─────────────────────────────────────────
st.markdown('<div class="header-title">🤖 بوت الأسئلة والأجوبة العربي</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">نظام ذكي للإجابة على الأسئلة باللغة العربية | TF-IDF + Cosine Similarity</div>', unsafe_allow_html=True)
st.divider()

# ─────────────────────────────────────────
# Layout: Two columns
# ─────────────────────────────────────────
col_context, col_chat = st.columns([1, 1], gap="large")

# ── LEFT: Context Input ──
with col_context:
    st.markdown("### 📄 أدخل النص العربي")
    context_input = st.text_area(
        label="النص",
        placeholder="الصق هنا النص العربي الذي تريد استخراج الإجابات منه...",
        height=300,
        label_visibility="collapsed"
    )

    top_n = st.slider("عدد الإجابات المقترحة", min_value=1, max_value=5, value=3)

    if st.button("✅ تحليل النص", use_container_width=True):
        if context_input.strip():
            sentences = split_sentences(context_input)
            if len(sentences) < 2:
                st.warning("النص قصير جداً. أضف المزيد من الجمل للحصول على نتائج أفضل.")
            else:
                vectors, idf = compute_tfidf(sentences)
                st.session_state.context_text = context_input
                st.session_state.sentences = sentences
                st.session_state.vectors = vectors
                st.session_state.idf = idf
                st.session_state.chat_history = []
                st.success(f"✅ تم تحليل {len(sentences)} جملة بنجاح!")
        else:
            st.error("الرجاء إدخال نص أولاً.")

    if st.session_state.sentences:
        st.caption(f"📊 الجمل المحللة: {len(st.session_state.sentences)}")

    # Sample text button
    if st.button("📌 تحميل نص تجريبي", use_container_width=True):
        sample = """المملكة العربية السعودية دولة عربية تقع في منطقة الشرق الأوسط.
تعتمد المملكة اعتماداً كبيراً على النفط كمصدر رئيسي للدخل القومي.
تسعى رؤية 2030 إلى تنويع مصادر الدخل وتقليل الاعتماد على النفط.
يشمل ذلك تطوير قطاعات السياحة والترفيه والتعدين والصناعة.
الرياض هي العاصمة السياسية والإدارية للمملكة العربية السعودية.
جدة تعتبر العاصمة الاقتصادية وميناء تجاري مهم على البحر الأحمر.
تأسست رؤية 2030 بمبادرة من ولي العهد الأمير محمد بن سلمان عام 2016.
يهدف البرنامج إلى رفع نسبة مشاركة المرأة في سوق العمل.
كما يهدف إلى تطوير قطاع التعليم وتحسين جودة الحياة للمواطنين.
صندوق الاستثمارات العامة أداة استثمارية محورية في تنفيذ رؤية 2030."""
        st.session_state["sample_loaded"] = sample
        st.rerun()

    if "sample_loaded" in st.session_state:
        sample_text = st.session_state.pop("sample_loaded")
        sentences = split_sentences(sample_text)
        vectors, idf = compute_tfidf(sentences)
        st.session_state.context_text = sample_text
        st.session_state.sentences = sentences
        st.session_state.vectors = vectors
        st.session_state.idf = idf
        st.session_state.chat_history = []
        st.success(f"✅ تم تحميل النص التجريبي وتحليل {len(sentences)} جملة!")

# ── RIGHT: Chat Interface ──
with col_chat:
    st.markdown("### 💬 اطرح سؤالك")

    # Chat history display
    if st.session_state.chat_history:
        for entry in st.session_state.chat_history:
            st.markdown(f'<div class="chat-bubble-user">❓ {entry["question"]}</div>', unsafe_allow_html=True)
            if entry["answers"]:
                for rank, (ans, conf) in enumerate(entry["answers"], 1):
                    bar_width = min(int(conf), 100)
                    st.markdown(f"""
                    <div class="answer-card">
                        <small style="color:#aaa;">الإجابة #{rank} — الثقة: {conf}%</small>
                        <div class="confidence-bar" style="width:{bar_width}%;"></div>
                        <p style="margin-top:10px;">{ans}</p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown('<div class="chat-bubble-bot">⚠️ لم أجد إجابة مناسبة في النص المقدم.</div>', unsafe_allow_html=True)
        st.divider()

    # Question input
    if st.session_state.sentences:
        question = st.text_input(
            "سؤالك",
            placeholder="اكتب سؤالك هنا باللغة العربية...",
            label_visibility="collapsed"
        )

        col_ask, col_clear = st.columns([3, 1])
        with col_ask:
            ask_btn = st.button("🔍 ابحث عن إجابة", use_container_width=True)
        with col_clear:
            clear_btn = st.button("🗑️ مسح", use_container_width=True)

        if ask_btn and question.strip():
            answers = get_top_answers(
                question,
                st.session_state.sentences,
                st.session_state.vectors,
                st.session_state.idf,
                top_n=top_n
            )
            st.session_state.chat_history.append({
                "question": question,
                "answers": answers,
                "timestamp": datetime.now().strftime("%H:%M:%S")
            })
            st.rerun()

        if clear_btn:
            st.session_state.chat_history = []
            st.rerun()

        # Export chat history
        if st.session_state.chat_history:
            output = io.StringIO()
            writer = csv.writer(output)
            writer.writerow(["الوقت", "السؤال", "الإجابة", "نسبة الثقة"])
            for entry in st.session_state.chat_history:
                for ans, conf in entry["answers"]:
                    writer.writerow([entry["timestamp"], entry["question"], ans, f"{conf}%"])
            st.download_button(
                label="📥 تصدير سجل المحادثة (CSV)",
                data=output.getvalue().encode("utf-8-sig"),
                file_name="chat_history.csv",
                mime="text/csv",
                use_container_width=True
            )
    else:
        st.info("👈 أدخل نصاً عربياً وحلله أولاً لبدء المحادثة.")

# ─────────────────────────────────────────
# Footer
# ─────────────────────────────────────────
st.divider()
st.markdown("""
<div style="text-align:center; color:#666; font-family:'Tajawal',sans-serif; font-size:13px;">
    بوت الأسئلة والأجوبة العربي | TF-IDF & Cosine Similarity | بناء بواسطة BaqarW-tech
</div>
""", unsafe_allow_html=True)
