# 🇮🇪 Irish News NLP – Topic Classification & Sentiment Analysis

A complete end‑to‑end NLP pipeline built to classify **Irish news articles** by **topic** and analyse their **sentiment** using NLTK VADER.

This project performs:
- 📦 Dataset extraction
- 🧹 Text preprocessing
- 🔠 TF‑IDF vectorisation
- 🤖 Model training (Linear SVM & Logistic Regression)
- 📊 Visual evaluation (confusion matrices, label distributions)
- 💬 Sentiment tagging using VADER
- 🔍 Topic prediction for custom text

---

## 📁 Project Structure
```
Irish_News_NLP_Classification/
│
├── Screenshot/
│   ├── 01_dataset_preview_head.png
│   ├── 02_article_length_distribution.png
│   ├── 03_training_label_distribution.png
│   ├── 04_linear_svm_confusion_matrix.png
│   ├── 05_model_comparison_summary.png
│   ├── 06_sentiment_distribution_by_topic.png
│   ├── 4_logistic_regression_confusion_matrix.png
│
├── IrishTimes_News_Dataset.zip
├── Irish_News_NLP_Topic_Classification_+_Sentiment.ipynb
└── irish_news_nlp_topic_classification_+_sentiment.py
```

---

## 🚀 How to Run

### 1️⃣ Install requirements
```bash
pip install nltk pandas numpy scikit-learn matplotlib seaborn
```

### 2️⃣ Download NLTK resources
Automatically handled in script:
```python
nltk.download('stopwords')
nltk.download('vader_lexicon')
```

### 3️⃣ Ensure dataset ZIP is in the main folder
`IrishTimes_News_Dataset.zip` — extracts automatically.

### 4️⃣ Run Python script
```bash
python irish_news_nlp_topic_classification_+_sentiment.py
```

### OR run notebook
```bash
jupyter notebook Irish_News_NLP_Topic_Classification_+_Sentiment.ipynb
```

---

## 🖼️ Visual Outputs

### 📌 Dataset Preview
![](Screenshot/01_dataset_preview_head.png)

### 📊 Article Length Distribution
![](Screenshot/02_article_length_distribution.png)

### 🏷️ Training Label Distribution
![](Screenshot/03_training_label_distribution.png)

### 🔷 Linear SVM Confusion Matrix
![](Screenshot/04_linear_svm_confusion_matrix.png)

### 🟧 Logistic Regression Confusion Matrix
![](Screenshot/4_logistic_regression_confusion_matrix.png)

### 📈 Model Comparison Summary
![](Screenshot/05_model_comparison_summary.png)

### 😊 Sentiment Distribution by Topic
![](Screenshot/06_sentiment_distribution_by_topic.png)

---

## 🔮 Topic Prediction Helper
```python
def predict_topic(text: str) -> str:
    cleaned = clean_text(text)
    vec = tfidf.transform([cleaned])
    pred = svm_model.predict(vec)[0]
    return pred
```

---

## 🎥 Demo Video (To Be Added)
*A placeholder for the final demo link.*

---

## ✍️ Author
**Abinash Prasana**
🇮🇪 NLP & AI Practice Project
