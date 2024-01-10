from flask import Flask, request, render_template_string
from predict.predict.run import TextPredictionModel
app = Flask(__name__)
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>StackOverflow Tags Prediction</title>
    <style>
        body {
            font-family: 'Arial', sans-serif;
            margin: 20px;
            text-align: center;
        }

        h2 {
            color: #333;
        }

        p {
            color: #666;
        }

        form {
            margin-top: 20px;
        }

        textarea {
            width: 80%;
            height: 100px;
            padding: 10px;
            margin-bottom: 10px;
        }

        button {
            padding: 10px 20px;
            background-color: #4CAF50;
            color: white;
            border: none;
            border-radius: 5px;
            cursor: pointer;
        }

        div {
            margin-top: 20px;
            padding: 10px;
            background-color: #f2f2f2;
            border-radius: 5px;
        }
    </style>
</head>
<body>
    <h2>StackOverflow Tag Prediction</h2>
    <p>The goal of the project is to develop a model that can automatically suggest relevant tags for a given question based on its content.</p>
    
    <form action="/" method="post">
        <textarea name="text" placeholder="Enter your question here..."></textarea>
        <br>
        <button type="submit">Predict Tag</button>
    </form>

    {% if predictions %}
    <div>
        <strong>Predictions:</strong> {{ predictions }}
    </div>
    {% endif %}
</body>
</html>
'''
@app.route('/', methods=['GET', 'POST'])
def index():
    predictions = None
    if request.method == 'POST':
        text_list = [request.form['text']]
        model = TextPredictionModel.from_artefacts('C:\\Users\\Frank Junior NGNIGUE\\Desktop\\poc-to-prod-capstone\\poc-to-prod-capstone\\train\\data\\artefacts')
        predictions = model.predict(text_list)
    return render_template_string(HTML_TEMPLATE, predictions=predictions)

if __name__ == '__main__':
    app.run(debug=True)