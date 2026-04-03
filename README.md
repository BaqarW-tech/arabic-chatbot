# 🤖 Arabic Q&A Bot | بوت الأسئلة والأجوبة العربي

A lightweight Arabic Question Answering chatbot built with **Streamlit and TF-IDF similarity**.

Users can paste Arabic text, ask a question about it, and the app returns the most relevant sentence from the text.

This version is optimized for **Streamlit Cloud deployment** and avoids heavy machine learning frameworks.

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
| QA Engine | TF-IDF Similarity |
| NLP Processing | Python Regex |
| Vectorization | scikit-learn |

---

# ⚙️ How It Works

1. User enters Arabic text
2. Text is split into sentences
3. TF-IDF vectors are created
4. Cosine similarity compares question with sentences
5. The most similar sentence is returned as the answer

This approach avoids heavy transformer models while still providing useful answers.

---

# 📂 Project Structure