from flask import Flask, request, render_template, Markup
import pickle
import numpy as np
import pandas as pd
from utils.disease import disease_dic
from utils.model import ResNet9
from PIL import Image
import io
import torch
from torchvision import transforms


# Initializing our flask application:
app = Flask(__name__)

# Loading our disease classification model:
disease_classes = ['Apple___Apple_scab',
                   'Apple___Black_rot',
                   'Apple___Cedar_apple_rust',
                   'Apple___healthy',
                   'Blueberry___healthy',
                   'Cherry_(including_sour)___Powdery_mildew',
                   'Cherry_(including_sour)___healthy',
                   'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
                   'Corn_(maize)___Common_rust_',
                   'Corn_(maize)___Northern_Leaf_Blight',
                   'Corn_(maize)___healthy',
                   'Grape___Black_rot',
                   'Grape___Esca_(Black_Measles)',
                   'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
                   'Grape___healthy',
                   'Orange___Haunglongbing_(Citrus_greening)',
                   'Peach___Bacterial_spot',
                   'Peach___healthy',
                   'Pepper,_bell___Bacterial_spot',
                   'Pepper,_bell___healthy',
                   'Potato___Early_blight',
                   'Potato___Late_blight',
                   'Potato___healthy',
                   'Raspberry___healthy',
                   'Soybean___healthy',
                   'Squash___Powdery_mildew',
                   'Strawberry___Leaf_scorch',
                   'Strawberry___healthy',
                   'Tomato___Bacterial_spot',
                   'Tomato___Early_blight',
                   'Tomato___Late_blight',
                   'Tomato___Leaf_Mold',
                   'Tomato___Septoria_leaf_spot',
                   'Tomato___Spider_mites Two-spotted_spider_mite',
                   'Tomato___Target_Spot',
                   'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
                   'Tomato___Tomato_mosaic_virus',
                   'Tomato___healthy']

disease_model_path = 'Models/plant_disease_model.pth'
disease_model = ResNet9(3, len(disease_classes))
disease_model.load_state_dict(torch.load(
    disease_model_path, map_location=torch.device('cpu')))
disease_model.eval()

# Loading our model:
model = pickle.load(open("Models/RFmodel.pkl", "rb"))

# Creating a function to predict image:
def predict_image(img, model = disease_model):
    """
    :params: image
    :return: prediction
    """
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
    ])
    
    image = Image.open(io.BytesIO(img))
    img_t = transform(image)
    img_u = torch.unsqueeze(img_t, 0)

    # Get predictions from model
    yb = model(img_u)
    # Pick index with highest probability
    _, preds = torch.max(yb, dim=1)
    prediction = disease_classes[preds[0].item()]
    # Retrieve the class label
    return prediction

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/crop-recommend")
def crop_recommend():
    return render_template('crop-recommend.html')


@app.route("/crop-predict", methods=["GET", "POST"])
def crop_prediction():
    if request.method == "POST":

        # Nitrogen
        nitrogen = float(request.form["nitrogen"])

        # Phosphorus
        phosphorus = float(request.form["phosphorus"])

        # Potassium
        potassium = float(request.form["potassium"])

        # Temperature
        temperature = float(request.form["temperature"])

        # Humidity Level
        humidity = float(request.form["humidity"])

        # PH level
        phLevel = float(request.form["ph-level"])

        # Rainfall
        rainfall = float(request.form["rainfall"])

        # Making predictions from the values:
        predictions = model.predict([[nitrogen, phosphorus, potassium, temperature, humidity, phLevel, rainfall]])

        output = predictions[0]
        finalOutput = output.capitalize()

        if (output == "rice" or output == "blackgram" or output == "pomegranate" or output == "papaya"
                or output == "cotton" or output == "orange" or output == "coffee" or output == "chickpea"
                or output == "mothbeans" or output == "pigeonpeas" or output == "jute" or output == "mungbeans"
                or output == "lentil" or output == "maize" or output == "apple"):
            cropStatement = finalOutput + " should be harvested. It's a Kharif crop, so it must be sown at the beginning of the rainy season e.g between April and May."


        elif (
                output == "muskmelon" or output == "kidneybeans" or output == "coconut" or output == "grapes" or output == "banana"):
            cropStatement = finalOutput + "should be harvested. It's a Rabi crop, so it must be sown at the end of " \
                                          "monsoon and beginning of winter season e.g between September and October. "

        elif output == "watermelon":
            cropStatement = finalOutput + "should be harvested. It's a Zaid Crop, so it must be sown between the " \
                                          "Kharif and rabi season i.e between March and June. "

        elif (output == "mango"):
            cropStatement = finalOutput + "should be harvested. It's a cash crop and also perennial. So you can grow " \
                                          "it anytime. "

    return render_template('cropResult.html', prediction_text=cropStatement)

@app.route("/disease-predict", methods = ['GET', 'POST'])
def disease_predict():
    title = 'Harvestify - Disease Detection'

    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files.get('file')
        if not file:
            return render_template('disease.html', title=title)
        try:
            img = file.read()

            prediction = predict_image(img)

            prediction = Markup(str(disease_dic[prediction]))
            return render_template('disease-result.html', prediction=prediction, title=title)
        except:
            pass
    return render_template('disease.html', title=title)


if __name__ == '__main__':
    app.run(debug=True)
