# 🤖 Arabic Q&A Bot | بوت الأسئلة والأجوبة العربي
 
An Arabic-language Question Answering chatbot built with Hugging Face
Transformers and Streamlit. Paste any Arabic text, ask a question about
it, and get an instant answer — powered by a fine-tuned AraBERT model.
 
## 🚀 Live Demo
Deploy on Streamlit Cloud. Requires a GPU-enabled tier for fast inference.
 
## 🛠️ Tech Stack
| Component      | Tool                                      |
|----------------|-------------------------------------------|
| NLP Model      | m3hrdadfi/bert-base-arabic-qa             |
| Preprocessing  | aubmindlab/AraBERT preprocessor           |
| Web UI         | Streamlit                                 |
| Tunnel (Colab) | pyngrok                                   |
 
## ⚙️ Setup & Run
 
### 1. Clone the repo
git clone https://github.com/BaqarW-tech/arabic-chatbot.git
cd arabic-chatbot
 
### 2. Install dependencies
pip install -r requirements.txt
 
### 3. Run the Streamlit app
streamlit run app.py
 
### Run in Google Colab
Open the notebook and run all cells.
pyngrok will create a public tunnel to the Streamlit app.
 
## 📂 Project Structure
arabic-chatbot/
├── app.py                # Streamlit web app
├── requirements.txt      # Python dependencies
├── .python-version       # Pins Python 3.11 for Streamlit Cloud
├── sample_questions.csv  # Example Arabic context/question pairs
└── README.md
 
## 📊 Sample Data
sample_questions.csv contains example Arabic context/question pairs.
Columns: context, question, expected_answer.
 
## 🌍 Use Case
Built as part of a data analytics portfolio targeting Arabic NLP
applications in the Saudi market. The model is fine-tuned for Arabic
reading comprehension using the ArTyDiQA dataset.
 
## 🐛 Known Issue (Fixed)
tokenizers==0.19.1 fails on Python 3.14 (Streamlit Cloud default as of
April 2026) due to a PyO3 version cap. Fixed by upgrading to
tokenizers>=0.21.0. See requirements.txt.
 
## 📄 License
MIT License

