<div align="center">

# 🇮🇪 <span style="font-size:2.4rem;">Irish News NLP – Topic Classification & Sentiment Analysis</span>

<p>
  <img src="https://img.shields.io/badge/Python-3.8%2B-3776AB?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/NLP-Text%20Processing-800080?logo=semanticweb&logoColor=white" alt="NLP Text Processing">
  <img src="https://img.shields.io/badge/Vectorizer-TFIDF-FFC300?logo=apachespark&logoColor=black" alt="TFIDF">
  <img src="https://img.shields.io/badge/Model-LinearSVM-FF8C42?logo=opsgenie&logoColor=white" alt="Linear SVM">
  <img src="https://img.shields.io/badge/Model-LogisticRegression-FF5733?logo=scikitlearn&logoColor=white" alt="Logistic Regression">
  <img src="https://img.shields.io/badge/Sentiment-VADER-28A745?logo=numpy&logoColor=white" alt="VADER">
  <img src="https://img.shields.io/badge/Project-Completed-success?logo=github&logoColor=white" alt="Status Completed">
</p>

</div>

---

## 🧠 Project Overview

This project builds an end‑to‑end **NLP classification + sentiment analysis** pipeline for Irish news articles.  
It uses the *Irish Times News Dataset* and walks through:

- 🧹 Robust text cleaning  
- 🔡 TF‑IDF feature extraction  
- 🤖 Topic classification using **Linear SVM** and **Logistic Regression**  
- 🙂 Sentiment tagging with **NLTK VADER**  
- 📊 Visual evaluation (confusion matrices, label distributions, sentiment plots)  

---

## ⚙️ Workflow Highlights

<details>
<summary><b>📦 Data & Pre‑processing</b></summary>

- Reads raw text files from the extracted ZIP  
- Normalises text to lowercase  
- Removes URLs, numbers, punctuation  
- Filters out English stopwords  
- Stores both **raw** and **cleaned** text in pandas DataFrames  

</details>

<details>
<summary><b>🧮 Feature Engineering</b></summary>

- Uses `TfidfVectorizer` with:
  - `max_features = 20000`
  - `ngram_range = (1, 2)` (uni‑grams + bi‑grams)
  - `min_df = 2` to drop very rare tokens  
- Transforms train and test sets into sparse TF‑IDF matrices  

</details>

<details>
<summary><b>🤖 Model Training & Evaluation</b></summary>

- Trains two models:
  - `LinearSVC()`  
  - `LogisticRegression(max_iter=1000, n_jobs=-1, random_state=42)`  
- Evaluates using:
  - `classification_report`
  - `confusion_matrix`
  - accuracy & weighted F1 score  
- Includes a compact **model comparison summary** for both models.  

</details>

<details>
<summary><b>💬 Sentiment Layer</b></summary>

- Uses `SentimentIntensityAnalyzer` from NLTK VADER  
- Assigns each article a label: **positive / neutral / negative**  
- Visualises:
  - sentiment counts per topic  
  - overall sentiment distribution  

</details>

<details>
<summary><b>🔍 Inference Utility</b></summary>

- Provides a helper to predict topic for any new text:

```python
def predict_topic(text: str) -> str:
    cleaned = clean_text(text)
    vec = tfidf.transform([cleaned])
    pred = svm_model.predict(vec)[0]
    return pred
```

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

These are already called inside the script / notebook:

```python
import nltk
nltk.download("stopwords")
nltk.download("vader_lexicon")
```

### 3️⃣ Make sure the dataset ZIP is available

Place `IrishTimes_News_Dataset.zip` in the project root.  
The script will automatically extract it.

### 4️⃣ Run the script

```bash
python irish_news_nlp_topic_classification_+_sentiment.py
```

### 5️⃣ Or open the notebook

```bash
jupyter notebook Irish_News_NLP_Topic_Classification_+_Sentiment.ipynb
```

---

## 🖼️ Screenshots & Visual Outputs

> The `Screenshot/` folder contains all generated plots.  
> Below is how they are used inside the README:

### 📌 Dataset Preview  
![](Screenshot/01_dataset_preview_head.png)

### 📊 Article Length Distribution  
![](Screenshot/02_article_length_distribution.png)

### 🏷️ Training Label Distribution  
![](Screenshot/03_training_label_distribution.png)

### 🔵 Linear SVM – Confusion Matrix  
![](Screenshot/04_linear_svm_confusion_matrix.png)

### 🟠 Logistic Regression – Confusion Matrix  
![](Screenshot/4_logistic_regression_confusion_matrix.png)

### 📈 Model Comparison Summary  
![](Screenshot/05_model_comparison_summary.png)

### 🙂 Sentiment Distribution by Topic  
![](Screenshot/06_sentiment_distribution_by_topic.png)

---

## 🎥 Demo Video (Placeholder)

> A section reserved for embedding the final demo link or GIF walkthrough.

---

## ✍️ Author

**Abinash Prasana**  
Irish‑based student exploring practical **NLP & Machine Learning** projects.

