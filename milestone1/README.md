# 📰 Intelligent News Credibility Analysis

A deployed Machine Learning system that classifies news articles as **Fake** or **Real** using TF-IDF feature extraction and an optimized Linear Support Vector Machine (LinearSVC).

Built as part of **Milestone 1 – Traditional Machine Learning & NLP**.

---

## 🚀 Live Demo

**Streamlit App:**  
https://news-credibility.streamlit.app

---

## 📌 What This Project Does

This application allows users to:

- Enter a news article manually  
- Upload a `.txt` file  
- Receive an instant Fake / Real prediction  
- View model confidence (decision margin)  
- Analyze text statistics (word count, unique words, frequency distribution)

The system uses a trained and serialized ML pipeline for fast inference.

---

## 🧠 Model Summary

- **Algorithm:** Linear Support Vector Machine (LinearSVC)  
- **Feature Extraction:** TF-IDF (Unigrams + Bigrams)  
- **Max Features:** 15,000  
- **Hyperparameter Tuning:** 3-Fold Cross Validation  
- **Evaluation Metric:** F1-Score  
- **Test Accuracy:** ~99%

---

## System Architecture and ML Pipeline Diagram
<img width="1000" height="auto" alt="image" src="https://github.com/user-attachments/assets/abe7d415-8103-42ef-b432-98e1735f8089" />

---
## 🖥️ Application Features

### Input Options
- Manual text entry  
- `.txt` file upload  

### Output Display
- Fake / Real classification  
- Decision margin score  
- Confidence interpretation  
- Word count  
- Unique word count  
- Character count  
- Top 10 most frequent words  
- Model configuration overview  

---

## 🗂 Project Structure

```
GenAI-capstone/
│
├── app/
│   └── app.py
│
├── data/
│   ├── raw/
│   │   └── raw_dataset.md
│   └── preprocessed/
│       ├── cleaned_dataset.csv
│       └── preprocess.py
│
├── model/
│   └── news_credibility_model.pkl
│
├── train_model.py
├── requirements.txt
└── README.md
```

---

## 📂 Dataset

Dataset used: **ISOT Fake News Dataset**

Due to GitHub file size limits, raw dataset files are not stored in this repository.

Download links are available in:

```
data/raw/raw_dataset.md
```

After downloading:

1. Place files inside `data/raw/`
2. Run preprocessing
3. Train the model

---

## ⚙️ Run Locally

### 1. Clone Repository

```bash
git clone https://github.com/your-username/GenAI-capstone.git
cd GenAI-capstone
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate     # Mac/Linux
venv\Scripts\activate        # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Start Application

```bash
streamlit run app/app.py
```

---

## ⚠️ Notes

- This milestone focuses strictly on traditional ML techniques.
- No Generative AI or LLMs are used.
- Model performance may vary on unseen external news domains.

---

## 👥 Team

- Khushi  
- Manpreet Singh  
- Avneet Singh  
- Riya Yadav  

---

## 📄 License

Developed for academic purposes (Milestone 1 – Traditional ML & NLP).
