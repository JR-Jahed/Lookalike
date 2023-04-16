from flask import Flask, request, render_template

from src.pipeline.predict_pipeline import PredictPipeline, CustomData

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict_api', methods=['POST'])
def predict_api():
    file = request.files['']

    custom_data = CustomData(file.stream.read())

    return PredictPipeline().predict(custom_data.image_batch)


@app.route('/predict', methods=['POST'])
def predict():

    file = request.files['file']

    custom_data = CustomData(file.stream.read())

    prediction = PredictPipeline().predict(custom_data.image_batch)

    predicted_class = prediction["class"]
    confidence = prediction["confidence"] * 100

    return render_template('home.html',
                           lookalike="Your lookalike is {}".format(predicted_class),
                           confidence="Confidence is: {} %".format(confidence))


if __name__ == "__main__":
    app.run()




















