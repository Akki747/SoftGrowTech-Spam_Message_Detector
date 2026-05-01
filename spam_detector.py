import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
# Step 1: Create dataset
data = {
    'message': [
        'Win money now',
        'Hello friend',
        'Claim your prize now',
        'How are you',
        'Free lottery ticket',
        'Call me now',
        'Congratulations you won',
        'Let us meet tomorrow',
        'Earn cash easily',
        'Are you coming today'
    ],
    'label': [
        'spam', 'ham', 'spam', 'ham', 'spam',
        'spam', 'spam', 'ham', 'spam', 'ham'
    ]
}

df = pd.DataFrame(data)

# Step 2: Convert text into numbers
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['message'])

# Step 3: Train model
model = MultinomialNB()
model.fit(X, df['label'])

print("📩 Spam Detector Ready! Type 'exit' to stop.\n")

# Step 4: User input loop
while True:
    user_msg = input("Enter message: ")

    if user_msg.lower() == "exit":
        print("Exiting Spam Detector...")
        break

    # Transform input
    msg_vector = vectorizer.transform([user_msg])

    # Predict
    prediction = model.predict(msg_vector)

    print("Prediction:", prediction[0].upper())
    print("-" * 30)
    print("Accuracy:", accuracy_score(df['label'], model.predict(X)))