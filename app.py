import streamlit as st
from transformers import pipeline
from arabert.preprocess import ArabertPreprocessor
 
st.set_page_config(page_title="Arabic Q&A Bot", page_icon="🤖")
 
st.title("🤖 أجب عن سؤالي")
st.markdown("اكتب نصاً عربياً، ثم اسألني سؤالاً عنه!")
 
@st.cache_resource
def load_model():
    prep = ArabertPreprocessor("aubmindlab/bert-base-arabertv02")
    qa = pipeline(
        "question-answering",
        model="m3hrdadfi/bert-base-arabic-qa",
        tokenizer="m3hrdadfi/bert-base-arabic-qa"
    )
    return prep, qa
 
with st.spinner("جاري تحميل النموذج... يرجى الانتظار"):
    preprocessor, qa_pipeline = load_model()
 
st.success("✅ النموذج جاهز! اكتب سؤالك الآن.")
 
context  = st.text_area("📄 النص (Context)", height=200,
                         placeholder="اكتب النص العربي هنا...")
question = st.text_input("❓ سؤالك (Question)",
                          placeholder="ماذا تريد أن تسأل؟")
 
if st.button("🤖 احصل على الجواب", type="primary"):
    if context and question:
        with st.spinner("🤔 جاري التفكير..."):
            try:
                clean_ctx = preprocessor.preprocess(context)
                clean_q   = preprocessor.preprocess(question)
                result    = qa_pipeline(question=clean_q, context=clean_ctx)
                st.success("✨ الإجابة:")
                st.markdown(f">>> {result['answer']}")
                st.caption(f"الثقة: {result['score']:.2%}")
            except Exception as e:
                st.error(f"حدث خطأ: {str(e)}")
    else:
        st.warning("⚠️ من فضلك اكتب النص والسؤال")
