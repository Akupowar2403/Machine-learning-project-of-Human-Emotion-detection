# Machine-learning-project-of-Human-Emotion-detection
Human emotion detection using audio and image  project is machine learning project in python language 
# Face Expression Detection Web Application

This is a web application built using Streamlit that allows users to upload an image, detects faces, and predicts emotions using a Convolutional Neural Network (CNN) model.

## Features
- **Face Detection**: Detects faces in the uploaded image using OpenCV's Haar Cascade Classifier.
- **Emotion Recognition**: Classifies the detected faces into different emotions such as Happy, Angry, Sad, Fear, Disgust, Neutral, and Surprise.
- **Real-Time Processing**: Upload an image, and the app will process it to detect faces and predict emotions.
- **User-Friendly Interface**: Built with Streamlit for easy interaction and image upload.
- **Progress Bars**: Displays progress while processing the uploaded image for face detection and emotion recognition.

## Prerequisites
To run this project locally, you need to have the following installed:
- Python 3.x
- Streamlit
- OpenCV
- Pillow
- NumPy

## Installation

1. Clone the repository to your local machine:

```bash
git clone https://github.com/yourusername/face-expression-detection.git
Navigate into the project directory:
bash
Copy code
cd face-expression-detection
Install the required Python libraries:
bash
Copy code
pip install -r requirements.txt
Download the required model files:

model.json
model_weights.h5
haarcascade_frontalface_default.xml
Make sure these files are placed in the appropriate directories (my_model/ and the root of the project).

Running the Application
Once the dependencies are installed and model files are set up, you can run the Streamlit app with the following command:

bash
Copy code
streamlit run app.py
This will start the Streamlit app and open it in your browser. You can then upload an image to detect faces and predict emotions.

How It Works
Upload an Image: The app allows you to upload an image in .jpg, .png, or .jpeg formats.
Face Detection: The app uses OpenCV to detect faces in the uploaded image.
Emotion Recognition: Each detected face is passed through a trained CNN model to predict the emotion.
Displaying Results: The app displays the image with rectangles around detected faces, labeled with the predicted emotions.
Model Overview
The model used for emotion prediction is a Convolutional Neural Network (CNN) that classifies emotions from facial expressions. The model is loaded using my_model/model.json and my_model/model_weights.h5.

Supported Emotions:
Happy
Angry
Sad
Fear
Disgust
Neutral
Surprise
About the Project
This project is a demonstration of facial expression recognition using a pre-trained CNN model. It leverages OpenCV for face detection and Streamlit for building the web interface.

Technologies Used:
Streamlit: For the web application interface.
OpenCV: For detecting faces in the uploaded images.
Pillow: For handling and displaying images.
NumPy: For image processing and model predictions.
TensorFlow/Keras: For loading the CNN model and making predictions.
Contributions
Feel free to fork the repository and contribute to it! Here are some ways you can contribute:

Report bugs or open issues.
Submit pull requests with improvements or new features.
License
This project is licensed under the MIT License - see the LICENSE file for details.

Acknowledgements
OpenCV for face detection algorithms.
Streamlit for creating the interactive web interface.
Keras and TensorFlow for building the CNN model.
