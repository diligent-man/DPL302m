from flask import Flask, render_template, request, jsonify
app = Flask(__name__, template_folder="./templates", static_folder="./static")

@app.route('/', methods=['GET','POST'])
def show_html():
    input_text = ''
    output_text = ''
    language = 'English'

    if request.method == 'POST':
        input_text = request.form.get('input_text')
        language = request.form.get('language', 'English')
        print(language)
        if language == "English":
            output_text = "This is model for English".upper()
        elif language == "Vietnamese":
            output_text = "This is model for Vietnamese".lower()

    return render_template('index1.html', output_text=output_text, input_text=input_text, lang=language)

@app.route('/about.html', methods=['GET','POST'])
def about_us():
    return render_template('about.html')

if __name__ == '__main__':
    app.run(debug=True)
