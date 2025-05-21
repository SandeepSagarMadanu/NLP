# NLP Text Analysis Project

This project demonstrates a basic NLP (Natural Language Processing) pipeline using Python and NLTK/Scikit-learn. It processes a collection of **US Presidential Inaugural Addresses**, provided in `corpus.txt`.

## 📂 Files

- `corpus.txt`: A collection of presidential inaugural addresses treated as a single text.
- `nlp.py`: Main script that performs preprocessing and feature extraction.

## 🧠 Features Implemented

1. **Data Loading**  
   Loads and reads the corpus from a plain text file.

2. **Tokenization**  
   Splits text into tokens using regular expressions.

3. **Stopword Removal**  
   Removes common English stopwords using NLTK's corpus.

4. **Stemming**  
   Applies PorterStemmer to reduce words to their base form.

5. **Lemmatization**  
   Uses WordNetLemmatizer for converting words to dictionary form.

6. **Bag of Words (BoW)**  
   Generates a BoW matrix using `CountVectorizer`.

7. **TF-IDF**  
   Computes term importance using `TfidfVectorizer`.

## 📦 Requirements

- Python 3.x
- `nltk`
- `scikit-learn`

Install the required libraries:

```bash
pip install nltk scikit-learn
