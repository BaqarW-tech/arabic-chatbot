# 🤖 Arabic Q&A Bot | بوت الأسئلة والأجوبة العربي

A lightweight Arabic Question Answering chatbot built using **Streamlit and TF-IDF similarity**.

Users can paste Arabic text, ask a question about it, and the system returns the most relevant sentence from the text.

This version avoids heavy machine learning frameworks and runs smoothly on **Streamlit Cloud**.

---

# 🚀 Live Demo

Deploy instantly on Streamlit Cloud.

No GPU required.

Startup time: **under 5 seconds**.

---

# 🛠 Tech Stack

| Component | Tool |
|---|---|
| Web UI | Streamlit |
| NLP Processing | Python + Regex |
| Sentence Ranking | TF-IDF |
| Similarity Metric | Cosine Similarity |
| Arabic Processing | Tashaphyne Stemmer |

---

# ⚙️ How It Works

1. User enters Arabic text
2. Text is cleaned and split into sentences
3. Words are stemmed using an Arabic stemmer
4. TF-IDF vectors are created
5. Cosine similarity compares the question with each sentence
6. The most relevant sentence is returned as the answer

This method is lightweight and suitable for cloud deployment.

---

# 📂 Project Structure