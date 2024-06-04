# model.py
import numpy as np
import tensorflow as tf
import google.generativeai as genai


class PlantDiseaseModel:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.class_names = [
            "Pepper_bell_Bacterial_spot", "Pepper_bell_healthy",
            "Tomato_Bacterial_spot", "Tomato_Early_blight", "Tomato_Late_blight",
            "Tomato_Leaf_Mold", "Tomato_Septoria_leaf_spot",
            "Tomato_Spider_mites_Two_spotted_spider_mite", "Tomato_Target_Spot",
            "Tomato_YellowLeaf_Curl_Virus", "Tomato_Tomato_mosaic_virus",
            "Tomato_healthy"
        ]
        self.interpreter = tf.lite.Interpreter(model_path=self.model_path)
        self.interpreter.allocate_tensors()

    def predict_image(self, image_path: str) -> str:
        image = tf.keras.preprocessing.image.load_img(image_path, target_size=(299, 299))
        input_arr = tf.keras.preprocessing.image.img_to_array(image)
        input_arr = np.expand_dims(input_arr, axis=0)
        input_arr = input_arr / 255.0  # Normalize the input image

        input_index = self.interpreter.get_input_details()[0]['index']
        self.interpreter.set_tensor(input_index, input_arr)
        self.interpreter.invoke()

        output_index = self.interpreter.get_output_details()[0]['index']
        output = self.interpreter.get_tensor(output_index)
        predicted_class_index = np.argmax(output)
        predicted_class_name = self.class_names[predicted_class_index]

        return predicted_class_name

def prompt_disease(disease: str) -> str:
    genai.configure(api_key="AIzaSyCBIF6CWfmv1QmpcCgQjdbPvvk-UrnCt5k")
    model = genai.GenerativeModel('gemini-pro')
    prompt = f"Jelaskan Penyakit {disease}: Pengertian, Penyebab, dan Cara Penanganan singkat dalam 3 paragraf."
    response = model.generate_content(prompt)
    return response.text
