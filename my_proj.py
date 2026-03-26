import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# ✅ Load dataset
file_path = r"D:\Documents\FA-KES-Dataset.csv"
df = pd.read_csv(file_path, encoding='latin1')

# ✅ Clean labels (handles 0/1 or FAKE/REAL)
df['labels'] = df['labels'].astype(str).str.lower()

# ✅ Select columns
X = df['article_title'].astype(str)   # input text
y = df['labels']                      # target

# ✅ Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ✅ Convert text to numbers (TF-IDF)
vectorizer = TfidfVectorizer(stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# ✅ Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

# ✅ Predict on test data
y_pred = model.predict(X_test_vec)

# ✅ Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy * 100:.2f}%")

# ✅ Custom input testing
while True:
    news = input("\nEnter news headline (or type 'exit'): ")
    if news.lower() == 'exit':
        break

    news_vec = vectorizer.transform([news])
    result = model.predict(news_vec)[0]

    # ✅ Convert output to Fake / Real
    if str(result).lower() in ["1", "real"]:
        print("Prediction: Real")
    else:
        print("Prediction: Fake")