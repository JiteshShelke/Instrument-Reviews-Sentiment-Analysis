# 🎵 Instrument Reviews Sentiment Analysis  

## 📌 Project Overview  
This project implements a **Sentiment Analysis System** for classifying customer reviews of musical instruments. Using **Natural Language Processing (NLP)** techniques and a **Logistic Regression Model**, the system predicts whether a review is **Positive, Negative, or Neutral**.  

The model is integrated into a **Flask web application**, enabling real-time sentiment classification.  

---

## 📊 Dataset  
The dataset contains customer reviews for musical instruments, with the following features:  

- `reviewerID`: Unique identifier for the reviewer  
- `asin`: Product ID  
- `reviewerName`: Name of the reviewer  
- `helpful`: Helpfulness rating from other users  
- `reviewText`: Full review text  
- `summary`: Short summary of the review  
- `overall`: Rating given by the reviewer (used for sentiment labeling)  
- `unixReviewTime`: Timestamp of the review  
- `reviewTime`: Human-readable date of the review  

### **Sentiment Labeling**  
- **Positive:** Rating > 3.0  
- **Negative:** Rating < 3.0  
- **Neutral:** Rating = 3.0  

The dataset is preprocessed by combining `reviewText` and `summary`, followed by text cleaning and transformation into numerical features.  

---

## 🛠️ Tools & Technologies Used  
### **Python Libraries:**  
- `pandas` – Data loading and manipulation  
- `nltk` – Text preprocessing (tokenization, stopword removal, lemmatization)  
- `matplotlib` – Data visualization  
- `scikit-learn` – ML model (Logistic Regression), TF-IDF vectorizer, label encoding  
- `imbalanced-learn (SMOTE)` – Handling class imbalance  
- `pickle` – Saving and loading the trained model  

### **Machine Learning & NLP Techniques:**  
- **Text Cleaning:** Removing punctuation, numbers, and special characters  
- **Tokenization & Lemmatization:** Converting words to their root forms  
- **TF-IDF Vectorization:** Transforming text data into numerical format  
- **SMOTE:** Handling class imbalance  
- **Logistic Regression:** Used for sentiment classification  

---

## 🌐 Web Application  
A **Flask**-based web application is developed to classify customer reviews in real-time.  

### **Features:**  
👉 Simple UI for review submission  
👉 Real-time sentiment classification  
👉 Scalable and deployable  

### **Implementation:**  
- **Backend:** Flask (Python)  
- **Frontend:** HTML, CSS, JavaScript  
- **Model Deployment:** The trained model, vectorizer, and label encoder are loaded using `pickle`.  

---

## 🚀 How to Run the Project  

### **1⃣ Clone the Repository**  
```bash
git clone https://github.com/jiteshshelke/Instrument-Reviews-Sentiment-Analysis.git
cd Instrument-Reviews-Sentiment-Analysis
```

### **2⃣ Install Dependencies**  
```bash
pip install -r requirements.txt
```

### **3⃣ Run the Flask Application**  
```bash
python app.py
```
The application will run at `http://127.0.0.1:5000/`

---

## 📌 Folder Structure  
```
Instrument-Reviews-Sentiment-Analysis/
│── dataset/                  # Raw dataset files  
│── models/                   # Saved ML model, TF-IDF vectorizer, label encoder  
│── static/                   # CSS, JavaScript files  
│── templates/                # HTML templates (Flask frontend)  
│── app.py                    # Flask web application  
│── preprocess.py             # Data preprocessing script  
│── train_model.py            # Machine learning model training  
│── README.md                 # Project documentation  
│── requirements.txt          # Python dependencies  
```

---

## 📊 Model Performance  
The trained Logistic Regression model is evaluated using:  
- **Accuracy**  
- **Precision, Recall, and F1-score**  
- **Confusion Matrix**  

Results indicate that the model effectively classifies sentiments, with improvements possible using deep learning techniques.  

---

## 📌 Future Enhancements  
👉 Improve performance with deep learning models (LSTMs, BERT)  
👉 Deploy the app on cloud platforms (Heroku, AWS)  
👉 Integrate with APIs for real-time review analysis  

---

## 📜 License  
This project is open-source and available under the **MIT License**.  

---

## 👨‍💻 Author  
**Jitesh Santosh Shelke**  
📌 **M.Sc. Data Analytics (Part II)**  
🏠 **Pillai College of Arts, Commerce & Science (Autonomous), New Panvel**  

---

🌟 **If you find this project useful, please ⭐ star the repository!** ⭐  

