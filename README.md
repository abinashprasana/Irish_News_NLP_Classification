<div align="center">

# 🇮🇪 <span style="font-size:2.4rem;">Irish News NLP – Topic Classification & Sentiment Analysis</span>

<p>
  <img src="https://img.shields.io/badge/Python-3.8%2B-3776AB?logo=python&logoColor=white">
  <img src="https://img.shields.io/badge/NLP-Text%20Processing-800080?logo=semanticweb&logoColor=white">
  <img src="https://img.shields.io/badge/Vectorizer-TFIDF-FFC300?logo=apachespark&logoColor=black">
  <img src="https://img.shields.io/badge/Model-LinearSVM-FF8C42?logo=opsgenie&logoColor=white">
  <img src="https://img.shields.io/badge/Model-LogisticRegression-FF5733?logo=scikitlearn&logoColor=white">
  <img src="https://img.shields.io/badge/Sentiment-VADER-28A745?logo=numpy&logoColor=white">
  <img src="https://img.shields.io/badge/Project-Completed-success?logo=github&logoColor=white">
</p>

</div>

---

## 🧠 Project Overview

This project builds an end-to-end **NLP classification + sentiment analysis** pipeline for Irish news articles.  
It uses the *Irish Times Topic Model Dataset* and performs:

- 🧹 Robust text cleaning  
- 🔡 TF-IDF feature extraction  
- 🤖 Topic classification using **Linear SVM** and **Logistic Regression**  
- 🙂 Sentiment tagging with **VADER**  
- 📊 Visual analysis of model accuracy, confusion matrices, and sentiment distribution  

---

## 📘 **Dataset Source & Credits**

🚨 **All dataset rights belong to the original creator.**  
This project simply uses the dataset for academic and learning purposes.

### 📌 **Dataset Used**  
**Irish Times Dataset for Topic Modeling**  
Hosted on **Kaggle** by: **manhoodz49**

📎 **Source Link:**  
👉 https://kaggle.com/datasets/manhoodz49/irish-times-dataset-for-topic-model

### 📄 Dataset Description (From the Author)

This dataset contains publicly available news articles from **The Irish Times**, collected to explore topic structures, perform text modeling, and analyse Ireland’s news landscape across categories like:

- Politics  
- Culture  
- Economy  
- Sports  
- Social Affairs  
- And more  

All content belongs to The Irish Times and the Kaggle dataset publisher.

---

## ⚙️ Workflow Highlights

<details>
<summary><b>📦 Data & Pre-processing</b></summary>

- Loads dataset from ZIP  
- Converts raw text into structured DataFrames  
- Cleans the data using custom functions  
</details>

<details>
<summary><b>🧮 Feature Engineering</b></summary>

- TF-IDF with 20k max features  
- Uses uni-grams + bi-grams (1,2)  
</details>

<details>
<summary><b>🤖 Model Training & Evaluation</b></summary>

- Trains Linear SVM & Logistic Regression  
- Generates classification reports  
- Confusion matrices  
- Weighted F1 scores  
</details>

<details>
<summary><b>💬 Sentiment Layer</b></summary>

- VADER-based tagging  
- Topic-wise sentiment analysis  
- Bar charts for sentiment distribution  
</details>

---

## 📁 Project Structure

```text
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

### 1️⃣ Install dependencies

```bash
pip install nltk pandas numpy scikit-learn matplotlib seaborn
```

### 2️⃣ Download NLTK resources

```python
import nltk
nltk.download("stopwords")
nltk.download("vader_lexicon")
```

### 3️⃣ Ensure the dataset ZIP is placed correctly

### 4️⃣ Run the script

```bash
python irish_news_nlp_topic_classification_+_sentiment.py
```

---

## 🖼️ Screenshots

### 📌 Dataset Preview  
![](Screenshot/01_dataset_preview_head.png)

### 📊 Article Length Distribution  
![](Screenshot/02_article_length_distribution.png)

### 🏷️ Label Distribution  
![](Screenshot/03_training_label_distribution.png)

### 🔵 Linear SVM Confusion Matrix  
![](Screenshot/04_linear_svm_confusion_matrix.png)

### 🟠 Logistic Regression Confusion Matrix  
![](Screenshot/4_logistic_regression_confusion_matrix.png)

### 📈 Model Comparison Summary  
![](Screenshot/05_model_comparison_summary.png)

### 🙂 Sentiment Distribution by Topic  
![](Screenshot/06_sentiment_distribution_by_topic.png)

---
## 🎥 Demo Video

https://github.com/user-attachments/assets/76952b18-e6eb-45a3-835c-3aa7294bfc2f

---

## ✍️ Author

**Abinash Prasana**  

