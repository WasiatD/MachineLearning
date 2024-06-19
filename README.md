# Plant Disease Classification using Inception V3 and TensorFlow Hub

This repository contains a machine learning project for classifying plant diseases using the Inception V3 pre-trained model and TensorFlow Hub. The project is implemented in Python and uses Google Colab as the execution environment.

## Dataset
The project uses the PlantVillage dataset, which is available at the [provided Google Drive link](https://drive.google.com/drive/folders/1--Y5wtswaXc4c_iDJzI1-NejJ6_7xJSn?usp=drive_link). The dataset contains images of various plant leaves with different disease conditions.

## Requirements
The following libraries can be installed using pip:
1. pip install numpy # linear algebra
2. pip install pandas # data processing, CSV file I/O (e.g. pd.read_csv)
3. pip install tensorflow
4. pip install opencv-python
5. pip install matplotlib
6. pip install tensorflow-hub
7. pip install google-generativeai

Additionally, the project uses the Inception V3 pre-trained model from TensorFlow Hub, which can be accessed at the [provided URL](https://www.kaggle.com/models/google/inception-v3/tensorFlow2/tf2-preview-feature-vector).

## Methodology
The project follows these steps:

1. Load the PlantVillage dataset from the provided Google Drive link.
2. Preprocess the images using the ImageDataGenerator from TensorFlow.
3. Load the Inception V3 pre-trained model from TensorFlow Hub and use it as the base model.
4. Perform transfer learning by adding a new classification layer on top of the pre-trained model.
5. Train the model using the preprocessed dataset.
6. Evaluate the model's performance on a validation set.
7. Use the Gemini API to obtain information about the identified plant diseases.

## Usage
To use this project, you can follow these steps:

1. Access the Google Colab environment.
2. Clone the repository or upload the necessary files.
3. Run the provided code cells to execute the plant disease classification pipeline.
4. Observe the model's performance and the disease information provided by the Gemini API.

## Acknowledgements
This project was developed using the resources provided by TensorFlow Hub and the PlantVillage dataset.

