from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import requests
from bs4 import BeautifulSoup

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Preprocess text
def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text

# Load and prepare dataset
def load_and_prepare_data():
    try:
        with open('dataset.json', 'r', encoding='utf-8') as file:
            data = json.load(file)['dataset']
        
        questions = [preprocess(item['question']) for item in data]
        answers = [item['answer'] for item in data]
        
        vectorizer = TfidfVectorizer()
        question_vectors = vectorizer.fit_transform(questions)
        
        return {
            'vectorizer': vectorizer,
            'question_vectors': question_vectors,
            'questions': questions,
            'answers': answers,
            'original_data': data
        }
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

# Global variable to store prepared data
prepared_data = load_and_prepare_data()

# Find best matching response
def find_response(user_input):
    if not prepared_data:
        return "Erreur de chargement des données"
    
    processed_input = preprocess(user_input)
    input_vector = prepared_data['vectorizer'].transform([processed_input])
    
    similarities = cosine_similarity(input_vector, prepared_data['question_vectors'])[0]
    best_match_index = np.argmax(similarities)
    
    if similarities[best_match_index] > 0.3:
        return prepared_data['answers'][best_match_index]
    
    return "Désolé, je n'ai pas compris votre question. Veuillez réessayer."

# Scrape website for useful links
def scrape_website():
    try:
        url = "https://ihec.rnu.tn/fr"
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Retrieve the current max ID in the dataset
        current_ids = [item['id'] for item in prepared_data['original_data']]
        max_id = max(current_ids, default=0)

        links = []
        for a in soup.find_all('a', href=True):
            text = a.text.strip()
            href = a['href']
            if text and href:
                # Increment ID dynamically
                max_id += 1
                links.append({
                    "id": max_id,
                    "category": "Liens Utiles",
                    "question": f"Où puis-je trouver des informations sur {text} ?",
                    "answer": f"Consultez cette page: {href}"
                })
        
        return links
    except Exception as e:
        print(f"Error scraping website: {e}")
        return []

# Update dataset with new data
def update_dataset(new_data):
    try:
        with open('dataset.json', 'r+', encoding='utf-8') as file:
            data = json.load(file)
            data['dataset'].extend(new_data)
            file.seek(0)
            json.dump(data, file, ensure_ascii=False, indent=4)
        print("Dataset successfully updated.")
    except Exception as e:
        print(f"Error updating dataset: {e}")

# Save feedback from users
def save_feedback(message, feedback):
    feedback_entry = {
        "message": message,
        "feedback": feedback
    }

    try:
        with open('feedback.json', 'a', encoding='utf-8') as file:
            json.dump(feedback_entry, file)
            file.write('\n')
    except Exception as e:
        print(f"Error saving feedback: {e}")

@app.route("/")
def home():
    return "Chatbot is running!"

@app.route("/chat", methods=["POST"])
def chat():
    try:
        user_message = request.json.get("message")
        response = find_response(user_message)
        return jsonify({"response": response})
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({"response": "Une erreur est survenue"}), 500

@app.route("/feedback", methods=["POST"])
def feedback():
    try:
        data = request.json
        message = data.get("message")
        user_feedback = data.get("feedback")

        if not message or not user_feedback:
            return jsonify({"error": "Invalid data"}), 400

        save_feedback(message, user_feedback)
        return jsonify({"status": "success", "message": "Feedback received"})
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({"error": "An error occurred"}), 500

@app.route("/scrape-and-update", methods=["POST"])
def scrape_and_update():
    try:
        new_data = scrape_website()
        if new_data:
            update_dataset(new_data)
            # Reload the dataset for updated responses
            global prepared_data
            prepared_data = load_and_prepare_data()
            return jsonify({"message": "Dataset updated with new links.", "new_data": new_data})
        else:
            return jsonify({"message": "No new data found during scraping."})
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({"message": "Une erreur est survenue lors du scraping."}), 500

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=8000)