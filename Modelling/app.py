# Install & run in Linux
# wget https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-linux-amd64.tgz --no-check-certificate
# tar -zxvf ngrok-v3-stable-linux-amd64.tgz
# mv ngrok /usr/bin/ngrok
# chmod 755 /usr/bin/ngrok
# https://stackoverflow.com/questions/72240708/cant-run-flask-on-ngrok
from flask import render_template
from base import SpellingCorrection
from flask_ngrok import run_with_ngrok
from flask import Flask, jsonify, request
from inference import Inferer


inferer = Inferer()
app = Flask(__name__, template_folder="./templates", static_folder="./static")
app.secret_key = "2YD4FL8vTYlImTY2EOBFib7CGSF_2anK73yjwLaz49fYuoxux"
# run_with_ngrok(app=app) # for gg colab


@app.route('/run_model', methods=['POST'])
def run_model():
    if request.method == 'POST':
        data = request.json
        text = data["text"]
        print(text)

        answer = inferer.infer_from_text(text)
        # selected_language = data["lang"]
        # if selected_language == "Vietnamese":
        #     result = vietnam_model.predict(input_data)
        # elif selected_language == "English":
        #     result = english_model.predict(input_data)
        # else:
        #     "Unsupported language", 400
    return jsonify({
        "output": answer
    }), 200


@app.route('/', methods=['GET','POST'])
def root():
    return render_template('index.html')


@app.route('/about.html', methods=['GET','POST'])
def about_us():
    return render_template('about.html')


if __name__ == '__main__':
    app.run(port=5000, debug=True)