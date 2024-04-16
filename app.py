from flask import Flask, render_template, request
from transformers import pipeline

app = Flask(__name__)
pipe = pipeline("text-classification", model="lxyuan/distilbert-base-multilingual-cased-sentiments-student")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def submit():
    user_input = request.form['user_input']
    result = pipe(user_input)[0]  # Perform sentiment analysis
    sentiment = result['label']   # Extract sentiment label
    return render_template('result.html', user_input=user_input, sentiment=sentiment)

if __name__ == '__main__':
    app.run(debug=True)
