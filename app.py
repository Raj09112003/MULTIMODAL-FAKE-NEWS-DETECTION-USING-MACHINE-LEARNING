import nltk
import os
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input

# Set a specific directory for NLTK data if needed
nltk_data_path = os.path.join(os.getcwd(), 'nltk_data')
if not os.path.exists(nltk_data_path):
    os.makedirs(nltk_data_path)

# Download necessary resources to the specified directory
nltk.download('punkt', download_dir=nltk_data_path)
nltk.download('punkt_tab', download_dir=nltk_data_path)
nltk.download('popular', download_dir=nltk_data_path)
nltk.download('stopwords', download_dir=nltk_data_path)

# Force NLTK to use the custom path
nltk.data.path.append(nltk_data_path)

from flask import Flask, request, jsonify, render_template, redirect, url_for
import numpy as np
from PIL import Image
import cv2
import joblib
import pickle

# Load models from pickle files
with open('text.pkl', 'rb') as file:
    model, vectorizer = pickle.load(open("text.pkl", "rb"))


with open('image.pkl', 'rb') as file:
    image_model = pickle.load(file)

video_model = joblib.load('video.pkl')

app = Flask(__name__)

# Text preprocessing function (as before)
def preprocess_text(text):
    import re
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    from nltk.stem import PorterStemmer

    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = word_tokenize(text)
    words = [word for word in words if word not in stopwords.words('english')]
    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in words]
    return ' '.join(words)


@app.route('/', methods=['GET', 'POST'])
def home():
    # Default empty values for predictions
    text_prediction = ""
    image_prediction = ""
    video_prediction = ""

    if request.method == 'POST':
        # Handle text input form submission
        if 'text_input' in request.form:
            user_text = request.form['text_input']
            processed_text = preprocess_text(user_text)
            vectorized_text = vectorizer.transform([processed_text])
            prediction = model.predict(vectorized_text)[0]
            text_prediction = 'Real News' if prediction == 1 else 'Fake News'

        # Handle image input form submission
        # Handle image input form submission (Unimodal - Image Only)
        import numpy as np
        from PIL import Image
        if 'image_input' in request.files:
            image = request.files['image_input']
            img = Image.open(image).convert('RGB')
            img = img.resize((224, 224))
            img = np.array(img) / 255.0  # Normalize pixel values
            img = np.expand_dims(img, axis=0)  # Shape: (1, 224, 224, 3)
            try:
                prediction = image_model.predict(img)[0]  # Assuming binary output
                image_prediction = 'Real Image' if prediction >= 0.5 else 'Fake Image'
            except Exception as e:
                image_prediction = f"Error in image prediction: {str(e)}"

        # Handle video input form submission
        # Handle video input form submission
        if 'video_input' in request.files:
            resnet_model = ResNet50(weights="imagenet", include_top=False, pooling="avg")

            video = request.files['video_input']
            video_path = "temp_video.mp4"
            video.save(video_path)

            

            cap = cv2.VideoCapture(video_path)
            frames = []
            success, frame = cap.read()

            while success:
                frame = cv2.resize(frame, (224, 224))
                frame = frame.astype('float32')
                frame = preprocess_input(np.expand_dims(frame, axis=0))
                features = resnet_model.predict(frame).flatten()
                frames.append(features)
                success, frame = cap.read()

            cap.release()

            if frames:
                video_features = np.mean(frames, axis=0)
                video_features = np.array(frames)
                video_features = video_features.reshape(video_features.shape[0], -1)  # reshape for prediction
                

                video_prediction = video_model.predict(video_features)[0]
                video_prediction = 'Real Video' if video_prediction >= 0.5 else 'Fake Video'


    return render_template(
        'index.html',
        text_prediction=text_prediction,
        image_prediction=image_prediction,
        video_prediction=video_prediction,
    )


if __name__ == '__main__':
    app.run(debug=True)
