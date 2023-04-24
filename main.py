import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model


def Predict(filename):
    
    #Load model
    tflite_model_path = "model/best_model_optimized.tflite"
        
    # Load TFLite model and allocate tensors.
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()
    
    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Test model on random input data.
    input_shape = input_details[0]['shape']
    input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)

    interpreter.invoke()
    
    output_data = interpreter.get_tensor(output_details[0]['index'])
    #print(output_data)
    
    SIZE = 180 #Resize to same size as training images
    img_path = 'static/images/'+filename
    img = np.asarray(Image.open(img_path).resize((SIZE,SIZE)))
    
    img = img/255.      #Scale pixel values
    
    img = np.expand_dims(img, axis=0)  #Get it tready as input to the network       
    
    pred = my_model.predict(img) #Predict                    
    
    #Convert prediction to class name
    #pred_class = le.inverse_transform([np.argmax(pred)])[0]
    print("Result is:", output_data)
    return  output_data #pred #pred_class
