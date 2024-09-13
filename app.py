from flask import Flask, render_template, request
# from llm import bug_fixing
from llmOpenAi import bug_fixing

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def index():
    resposta = ""
    if request.method == 'POST':
        text = request.form['userInput']
        resposta = bug_fixing(text)
    return render_template('index.html', resposta = resposta)

if __name__ == '__main__':
    app.run(debug=True)