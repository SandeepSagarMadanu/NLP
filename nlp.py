import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# Download required NLTK resources (run once)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# === Step 1: Load data from corpus.txt ===
with open('NLP\corpus.txt', 'r') as f:
    corpus = [line.strip() for line in f.readlines() if line.strip()]

print("Corpus Loaded:", corpus)

# === Step 2: Tokenization ===
print("\n--- Tokenization ---")
for sentence in corpus:
    tokenizer = RegexpTokenizer(r"\w+")
    tokens = tokenizer.tokenize(sentence)
    print(f"{sentence} => {tokens}")

# === Step 3: Stopword Removal ===
stop_words = set(stopwords.words('english'))
print("\n--- After Stopword Removal ---")
filtered_corpus = []
for sentence in corpus:
    tokenizer = RegexpTokenizer(r"\w+")
    tokens = tokenizer.tokenize(sentence)
    filtered = [w for w in tokens if w.lower() not in stop_words]
    filtered_corpus.append(filtered)
    print(f"{sentence} => {filtered}")

# === Step 4: Stemming ===
stemmer = PorterStemmer()
print("\n--- After Stemming ---")
for tokens in filtered_corpus:
    stemmed = [stemmer.stem(w) for w in tokens]
    print(f"{tokens} => {stemmed}")

# === Step 5: Lemmatization ===
lemmatizer = WordNetLemmatizer()
print("\n--- After Lemmatization ---")
for tokens in filtered_corpus:
    lemmatized = [lemmatizer.lemmatize(w) for w in tokens]
    print(f"{tokens} => {lemmatized}")

# === Step 6: Bag of Words ===
print("\n--- Bag of Words ---")
vectorizer_bow = CountVectorizer()
bow = vectorizer_bow.fit_transform(corpus)
print("Vocabulary:", vectorizer_bow.get_feature_names_out())
print("BoW Matrix:\n", bow.toarray())

# === Step 7: TF-IDF ===
print("\n--- TF-IDF ---")
vectorizer_tfidf = TfidfVectorizer()
tfidf = vectorizer_tfidf.fit_transform(corpus)
print("Vocabulary:", vectorizer_tfidf.get_feature_names_out())
print("TF-IDF Matrix:\n", tfidf.toarray())
