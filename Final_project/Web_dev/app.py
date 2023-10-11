# from flask import Flask, render_template, request, redirect
#
# app = Flask(__name__)
#
# @app.route('/')
# def show_html():
#     return render_template('frontend.html')
#
# if __name__ == '__main__':
#     app.run(debug=True)


from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def show_html():
    freeform2_data = ''
    freeform1_data = ''
    language = 'English'
    if request.method == 'POST':
        # Lấy dữ laiệu từ biểu mẫu nếu là phương thức POST
        freeform1_data = request.form.get('freeform1')
        language = request.form.get('language', 'English')
        print(language)
        if language == "English":
            freeform2_data = "This is model for English"
        elif language == "Vietnamese":
            freeform2_data = "This is model for Vietnamese"

    return render_template('index.html', freeform2=freeform2_data, freeform1=freeform1_data, lang = language)



if __name__ == '__main__':
    app.run(debug=True)
