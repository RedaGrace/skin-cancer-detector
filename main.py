import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model


def Predict(filename):
    
    #Load model
    tflite_model_path = "model/best_model_optimized.tflite"
    my_model = load_model(tflite_model_path)
    
    # Load TFLite model and allocate tensors.
    with open(tflite_model_file, 'rb') as fid:
        tflite_model = fid.read()
    
    SIZE = 180 #Resize to same size as training images
    img_path = 'static/images/'+filename
    img = np.asarray(Image.open(img_path).resize((SIZE,SIZE)))
    
    img = img/255.      #Scale pixel values
    
    img = np.expand_dims(img, axis=0)  #Get it tready as input to the network       
    
    pred = my_model.predict(img) #Predict                    
    
    #Convert prediction to class name
    #pred_class = le.inverse_transform([np.argmax(pred)])[0]
    print("Result is:", pred)
    return  pred #pred_class
