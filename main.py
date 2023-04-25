import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model

labels = ['actinic keratosis', 'basal cell carcinoma', 'dermatofibroma', 'melanoma', 'nevus', 
          'pigmented benign keratosis', 'seborrheic keratosis', 'squamous cell carcinoma', 'vascular lesion']

class TensorflowLiteClassificationModel:
    def __init__(self, model_path, labels, image_size=180):
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self._input_details = self.interpreter.get_input_details()
        self._output_details = self.interpreter.get_output_details()
        self.labels = labels
        self.image_size=image_size

    def run_from_filepath(self, image_path):
        input_data_type = self._input_details[0]["dtype"]
        image = np.array(Image.open(image_path).resize((self.image_size, self.image_size)), dtype=input_data_type)
        #if input_data_type == np.float32:
        image = image / 255.    #Scale pixel values

        #if image.shape == (1, 180, 180):
        image = np.expand_dims(image, axis=0)

        return self.run(image)

    def run(self, image):
        """
        args:
          image: a (1, image_size, image_size, 3) np.array

        Returns list of [Label, Probability], of type List<str, float>
        """

        self.interpreter.set_tensor(self._input_details[0]["index"], image)
        self.interpreter.invoke()
        tflite_interpreter_output = self.interpreter.get_tensor(self._output_details[0]["index"])
        probabilities = np.array(tflite_interpreter_output[0])

        # create list of ["label", probability], ordered descending probability
        label_to_probabilities = []
        for i, probability in enumerate(probabilities):
            label_to_probabilities.append([self.labels[i], float(probability)])
        return sorted(label_to_probabilities, key=lambda element: element[1])


def Predict(filename):
    
    #Load model
    tflite_model_path = "model/best_model_optimized.tflite"
   
    img_path = 'static/images/'+filename
    
    # Usage
    model = TensorflowLiteClassificationModel(tflite_model_path, labels)
    (label, probability) = model.run_from_filepath(img_path)                   
    
    #Convert prediction to class name
    #pred_class = le.inverse_transform([np.argmax(pred)])[0]
    print("Result is:", (label, probability))
    return  (label, probability) #pred #pred_class
