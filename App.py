from flask import Flask, request, jsonify
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import RandomOverSampler

app = Flask(__name__)

# --- Load the Model and Vectorizer ---
try:
    with open("bagging_model.pkl", "rb") as model_file:
        bagging_model = pickle.load(model_file)
    with open("vectorizer.pkl", "rb") as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)
    print("Model and vectorizer loaded successfully.")
except FileNotFoundError as e:
    print(f"Error loading model/vectorizer: {e}")
    # Train model and create pickle file if it doesn't exist
    # --- Data Preparation ---
    data = {
        'text': [
            "I want to book a flight to London", "Find me a hotel in Paris", "What are some attractions in Rome?",
            "Book a flight from New York to London for next Monday", "Show me hotels in Paris with a rating of 4 stars or higher",
            "What are some things to do in Rome?", "How do I get to the airport from here?", "Book a cab for me to the airport",
            "What's the weather like in London next week?", "Is it going to rain in Paris tomorrow?",
            "Recommend me a destination for beaches", "Recommend me a destination for mountains"  # Travel recommendation
        ],
        'intent': [
            "find_flights", "find_hotels", "find_attractions",
            "find_flights", "find_hotels",
            "find_attractions", "get_directions", "book_cab",
            "get_weather", "get_weather",
            "recommend_destination", "recommend_destination"
        ]
    }
    df = pd.DataFrame(data)

    # --- Text Preprocessing ---
    def preprocess_text(text):
        tokens = word_tokenize(text.lower())
        tokens = [word for word in tokens if word.isalnum()]
        tokens = [word for word in tokens if word not in stopwords.words('english')]
        return " ".join(tokens)

    df['cleaned_text'] = df['text'].apply(preprocess_text)
    df = df[df['cleaned_text'] != ""].reset_index(drop=True)

    # --- Feature Engineering ---
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df['cleaned_text'])
    y = df['intent']

    # --- Data Splitting ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # --- Handle Class Imbalance with Oversampling ---
    ros = RandomOverSampler(random_state=42)
    X_train_dense, y_train = ros.fit_resample(X_train.toarray(), y_train)

    # --- Model Training ---
    svc_model = SVC(kernel='linear', probability=True)
    bagging_model = BaggingClassifier(
        estimator=svc_model,
        n_estimators=10,
        random_state=42
    )

    bagging_model.fit(X_train_dense, y_train)
    y_pred = bagging_model.predict(X_test.toarray())

    # --- Evaluation ---
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Bagging Model Accuracy: {accuracy:.3f}")

    # --- Save the Model and Vectorizer ---
    with open("bagging_model.pkl", "wb") as model_file:
        pickle.dump(bagging_model, model_file)

    with open("vectorizer.pkl", "wb") as vectorizer_file:
        pickle.dump(vectorizer, vectorizer_file)

    print("Model and vectorizer saved successfully!")
    with open("bagging_model.pkl", "rb") as model_file:
        bagging_model = pickle.load(model_file)
    with open("vectorizer.pkl", "rb") as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)
except Exception as e:
    print(f"Unexpected error: {e}")

# --- Text Preprocessing function ---
def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalnum()]
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    return " ".join(tokens)

# --- Travel API Simulation (Replace with a real API!) ---
def get_travel_recommendations(intent, text):
    if intent == "find_flights":
        return "Simulated: Found flights from $200-$500 depending on date!"
    elif intent == "find_hotels":
        return "Simulated: Found hotels ranging from budget to luxury!"
    elif intent == "find_attractions":
        return "Simulated: Found many attractions - museums, parks, etc.!"
    elif intent == "get_weather":
        return "Simulated: Weather is sunny but bring an umbrella, just in case!"
    elif intent == "recommend_destination":
        return "Simulated: Recommending a tropical place!"
    else:
        return "Simulated: Sorry, I don't know how to help with that yet."

# --- Webhook Endpoint ---
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        text = data['text']

        # --- Preprocess the text ---
        cleaned_text = preprocess_text(text)

        # --- Transform the text using the loaded vectorizer ---
        text_vectorized = vectorizer.transform([cleaned_text])

        # --- Make the prediction ---
        intent = bagging_model.predict(text_vectorized.toarray())[0]

        # --- Get Travel Recommendations from API (Simulation) ---
        travel_recommendation = get_travel_recommendations(intent, text)

        # --- Prepare the response ---
        response = {
            'intent': intent,
            'travel_recommendation': travel_recommendation
        }

        return jsonify(response)

    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    try:
        nltk.data.find("corpora/wordnet")
    except LookupError:
        nltk.download('wordnet')
    nltk.download('punkt')
    nltk.download('stopwords')
    app.run(debug=True)
