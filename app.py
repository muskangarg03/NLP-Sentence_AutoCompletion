from flask import Flask, render_template, request, jsonify
import nltk
from nltk.tokenize import word_tokenize
from collections import defaultdict
import dill as pickle

app = Flask(__name__)

# Download NLTK data
nltk.download('punkt')

# Load the pickled model
model_file_path = "ngram_model.pkl"
with open(model_file_path, 'rb') as f:
    model = pickle.load(f)

# Function to predict the next words given a context
def predict_next_words(model, context):
    if context in model:
        return list(model[context].keys())
    else:
        return []

# Function to autocomplete a sentence
def autocomplete_sentence(model, n, sentence, max_words=5):
    tokens = word_tokenize(sentence.lower())
    context = tokens[-(n - 1):]
    for _ in range(max_words):
        next_words = predict_next_words(model, tuple(context))
        if not next_words or '</s>' in next_words:
            break
        next_word = next_words[0]  # Select the first word as the prediction
        sentence += ' ' + next_word
        context = context[1:] + [next_word]
    return sentence

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/autocomplete', methods=['POST'])
def autocomplete():
    input_sentence = request.form['input_sentence']
    tokens = word_tokenize(input_sentence.lower())
    context = tokens[-2:]  # Considering bigram context
    suggestions = predict_next_words(model, tuple(context))
    return jsonify({'suggestions': suggestions})

@app.route('/autocomplete_sentence', methods=['POST'])
def autocomplete_sentence_route():
    input_sentence = request.form['input_sentence']
    autocompleted_sentence = autocomplete_sentence(model, 3, input_sentence)
    return jsonify({'autocompleted_sentence': autocompleted_sentence})

if __name__ == '__main__':
    app.run(debug=True)
