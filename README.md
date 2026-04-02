# 🤖 Arabic Q&A Bot | بوت الأسئلة والأجوبة العربي

An Arabic-language Question Answering chatbot built with Hugging Face Transformers and Streamlit. Paste any Arabic text, ask a question about it, and get an instant answer — all powered by a fine-tuned AraELECTRA model.

---

## 🚀 Demo

| Input | Output |
|---|---|
| Arabic context paragraph | Extracted answer in Arabic |

---

## 🛠️ Tech Stack

| Component | Tool |
|---|---|
| NLP Model | [`wissamantoun/araelectra-base-artydiqa`](https://huggingface.co/wissamantoun/araelectra-base-artydiqa) |
| Preprocessing | [`aubmindlab/AraBERT`](https://github.com/aub-mind/arabert) preprocessor |
| Web UI | Streamlit |
| Tunnel (Colab) | pyngrok |

---

## ⚙️ Setup & Run

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/arabic-qa-bot.git
cd arabic-qa-bot
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit app
```bash
streamlit run app.py
```

### Run in Google Colab
Open `my_arabic_bot.ipynb` in [Google Colab](https://colab.research.google.com/) and run all cells. The notebook handles model loading and launches the app via pyngrok.

---

## 📂 Project Structure

```
arabic-qa-bot/
├── my_arabic_bot.ipynb   # Main Colab notebook
├── app.py                # Streamlit web app
├── requirements.txt      # Python dependencies
├── sample_questions.csv  # Example context/question pairs for testing
└── README.md
```

---

## 📊 Sample Data

`sample_questions.csv` contains example Arabic context/question pairs you can use to test the model. Columns: `context`, `question`, `expected_answer`.

---

## 🌍 Use Case

Built as part of a data analytics portfolio targeting Arabic NLP applications in the Saudi market. The model is specialized for Arabic reading comprehension using the ArTyDiQA dataset.

---

## 📄 License

MIT License
