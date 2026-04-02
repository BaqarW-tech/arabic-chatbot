import streamlit as st
from transformers import pipeline
from arabert.preprocess import ArabertPreprocessor

st.set_page_config(page_title="Arabic Q&A Bot", page_icon="🤖")

st.title("🤖 أجب عن سؤالي")
st.markdown("اكتب نصاً عربياً، ثم اسألني سؤالاً عنه!")

# Load the model (Cached so it doesn't reload every time)
@st.cache_resource
def load_model():
    prep = ArabertPreprocessor("aubmindlab/araelectra-base-discriminator")
    qa = pipeline("question-answering", model="wissamantoun/araelectra-base-artydiqa")
    return prep, qa

preprocessor, qa_pipeline = load_model()

# Input Boxes
context = st.text_area("📄 النص (Context)", height=200)
question = st.text_input("❓ سؤالك (Question)")

if st.button("احصل على الجواب"):
    if context and question:
        with st.spinner("بتفكر..."):
            clean_ctx = preprocessor.preprocess(context)
            clean_q = preprocessor.preprocess(question)
            result = qa_pipeline(question=clean_q, context=clean_ctx)
        
        st.success("✨ الإجابة:")
        st.markdown(f">>> {result['answer']}")
    else:
        st.warning("من فضلك اكتب النص والسؤال")
