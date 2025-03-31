import pickle
import string
import re
from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

# Download NLTK data if not already present
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# Initialize Flask app
app = Flask(__name__)

# Load model and TF-IDF vectorizer
with open('logistic_regression_model.pkl', 'rb') as f:
    Classifier = pickle.load(f)

with open('tfidf_vectorizer.pkl', 'rb') as f:
    TF_IDF = pickle.load(f)

# Define text processing functions
def Text_Cleaning(Text):
    Text = Text.lower()
    punc = str.maketrans(string.punctuation, ' '*len(string.punctuation))
    Text = Text.translate(punc)
    Text = re.sub(r'\d+', '', Text)
    Text = re.sub('https?://\S+|www\.\S+', '', Text)
    Text = re.sub('\n', '', Text)
    return Text

def Text_Processing(Text):
    Stopwords = set(stopwords.words('english'))
    Lemmatizer = WordNetLemmatizer()
    Tokens = word_tokenize(Text)
    Processed_Text = [Lemmatizer.lemmatize(word) for word in Tokens if word not in Stopwords]
    return " ".join(Processed_Text)

def predict_sentiment(new_review):
    new_review_cleaned = Text_Cleaning(new_review)
    new_review_processed = Text_Processing(new_review_cleaned)
    new_review_tfidf = TF_IDF.transform([new_review_processed])
    sentiment_code = Classifier.predict(new_review_tfidf)[0]
    
    # Map sentiment codes to labels
    if sentiment_code == 0:
        sentiment_label = 'Negative that is 0'
    elif sentiment_code == 1:
        sentiment_label = 'Neutral that is 1'
    elif sentiment_code == 2:
        sentiment_label = 'Positive that is 2'
    else:
        sentiment_label = 'Unknown'
    
    return sentiment_label

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        review = request.form['review']
        sentiment = predict_sentiment(review)
        return render_template('result.html', sentiment=sentiment)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
