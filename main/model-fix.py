import tensorflow as tf
from PIL import Image
import numpy as np

class plant_disease_model:
    def __init__(self, model_path):
        self.loaded_model = tf.saved_model.load(model_path)

    def predict_tf(self, img_path):
        class_names = [
            "Pepper__bell___Bacterial_spot", "Pepper__bell___healthy",
            "Strawberry___Leaf_scorch", "Strawberry___healthy",
            "Tomato_Bacterial_spot", "Tomato_Early_blight", "Tomato_Late_blight",
            "Tomato_Leaf_Mold", "Tomato_Septoria_leaf_spot",
            "Tomato_Spider_mites_Two_spotted_spider_mite", "Tomato__Target_Spot",
            "Tomato__Tomato_YellowLeaf__Curl_Virus", "Tomato__Tomato_mosaic_virus",
            "Tomato_healthy"
        ]

        # Load and preprocess the image
        img = Image.open(img_path)
        img = img.resize((299, 299))  # Resize the image to match model input size
        img_array = np.array(img)  # Convert image to numpy array
        img_array = img_array.astype(np.float32) / 255.0  # Normalize pixel values and convert to float32

        # Predict the class of the image using the loaded_model directly
        prediction = self.loaded_model(np.expand_dims(img_array, axis=0))

        # Get the predicted class index
        predicted_class_index = np.argmax(prediction)
        
        # Get the predicted class name
        predicted_class_name = class_names[predicted_class_index]

        return predicted_class_name

    def prompt_disease(self, disease):
        # Dapatkan kunci API Google Anda dari userdata
        GOOGLE_API_KEY = userdata.get('GOOGLE_API_KEY')

        # Konfigurasi kunci API
        genai.configure(api_key=GOOGLE_API_KEY)

        # Buat instance model GenerativeModel untuk Gemini
        model = genai.GenerativeModel('gemini-pro')

        # Prompt berdasarkan penyakit
        prompt = f"Jelaskan Penyakit {disease}: Pengertian, Penyebab, dan Cara Penanganan singkat dalam 3 paragraf."

        # Gunakan model untuk menghasilkan konten berdasarkan prompt
        response = model.generate_content(prompt)

        # Cetak hasil konten yang dihasilkan
        return response.text

    def main_tf(self, image_path):
        # Load the image and predict the class
        predicted_class_name = self.predict_tf(image_path)

        # Print the predicted class name
        print(self.prompt_disease(predicted_class_name))

# Usage example:
if __name__ == "__main__":
    model_path = "/content/model_fix"
    image_path = "/TomatoYellowCurlVirus1.JPG"
    plant_disease_model_instance = plant_disease_model(model_path)
    plant_disease_model_instance.main_tf(image_path)
