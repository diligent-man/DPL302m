from flask import Flask, render_template, request, jsonify
from model import model_en, model_vn

app = Flask(__name__, template_folder="./templates", static_folder="./static")
english_model = model_en.Model_en()
english_model.load_model()
vietnam_model = model_vn.Model_vn()
vietnam_model.load_model()

@app.route('/run_model', methods=['POST'])
def run_model():
    if request.method == 'POST':
        data = request.json
        input_data = data["text"]        
        selected_language = data["lang"]
        if selected_language == "Vietnamese":
            result = vietnam_model.predict(input_data)
        elif selected_language == "English":
            result = english_model.predict(input_data)
        else:
            "Unsupported language", 400
    return jsonify({
        "output": result 
    }), 200

@app.route('/', methods=['GET','POST'])
def root():
    # input_text = ''
    # output_text = ''
    # language = 'English'

    # if request.method == 'POST':
    #     input_text = request.form.get('input_text')
    #     language = request.form.get('language', 'English')
    #     print(language)
    #     if language == "English":
    #         output_text = "This is model for English".upper()
    #     elif language == "Vietnamese":
    #         output_text = "This is model for Vietnamese".lower()

    return render_template('index1.html')

@app.route('/about.html', methods=['GET','POST'])
def about_us():
    return render_template('about.html')

if __name__ == '__main__':
    app.run(debug=True, port=3000)
