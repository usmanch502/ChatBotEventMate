from flask import Flask, request, jsonify
import nltk
from nltk.stem import WordNetLemmatizer
import pickle
import numpy as np
from keras.models import load_model
import json
import random
from flask_cors import CORS
nltk.data.path.append('nltk_data')
app = Flask(__name__)
CORS(app)  # Enable CORS for Flutter

lemmatizer = WordNetLemmatizer()
model = load_model("chatbot_model.h5")
intents = json.loads(open("intents.json").read())
words = pickle.load(open("words.pkl", "rb"))
classes = pickle.load(open("classes.pkl", "rb"))

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words):
    sentence_words = clean_up_sentence(sentence)
    bag = [1 if w in sentence_words else 0 for w in words]
    return np.array(bag)

def predict_class(sentence, model):
    p = bow(sentence, words)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return [{"intent": classes[r[0]], "probability": str(r[1])} for r in results]

def get_response(ints, intents_json):
    if not ints:
        return "I'm sorry, I don't understand."
    tag = ints[0]["intent"]
    for i in intents_json["intents"]:
        if i["tag"] == tag:
            return random.choice(i["responses"])
    return "I'm sorry, I don't understand."

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    message = data['message']
    intents_list = predict_class(message, model)
    response = get_response(intents_list, intents)
    return jsonify({"response": response})

if __name__ == '__main__':
    app.run(debug=True)
