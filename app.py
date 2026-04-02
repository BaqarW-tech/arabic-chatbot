import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering
from arabert.preprocess import ArabertPreprocessor

st.set_page_config(page_title="Arabic Q&A Bot", page_icon="🤖")

st.title("🤖 أجب عن سؤالي")
st.markdown("اكتب نصاً عربياً، ثم اسألني سؤالاً عنه!")

MODEL_NAME = "wissamantoun/araelectra-base-artydiqa"
PREPROCESSOR_NAME = "aubmindlab/araelectra-base-discriminator"

@st.cache_resource
def load_model():
    prep = ArabertPreprocessor(PREPROCESSOR_NAME)
    # Load tokenizer and model explicitly to bypass pipeline task registry issues
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME)
    qa = pipeline(
        "question-answering",
        model=model,
        tokenizer=tokenizer
    )
    return prep, qa

try:
    preprocessor, qa_pipeline = load_model()
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error(f"❌ فشل تحميل النموذج: {e}")

# Input Boxes
context = st.text_area("📄 النص (Context)", height=200,
                        placeholder="مثال: الرياض هي عاصمة المملكة العربية السعودية...")
question = st.text_input("❓ سؤالك (Question)",
                          placeholder="مثال: ما هي عاصمة المملكة؟")

if st.button("احصل على الجواب", disabled=not model_loaded):
    if context and question:
        with st.spinner("بتفكر..."):
            clean_ctx = preprocessor.preprocess(context)
            clean_q = preprocessor.preprocess(question)
            result = qa_pipeline(question=clean_q, context=clean_ctx)

        st.success("✨ الإجابة:")
        st.markdown(f"**{result['answer']}**")
        st.caption(f"درجة الثقة: {result['score']:.2%}")
    else:
        st.warning("من فضلك اكتب النص والسؤال")
