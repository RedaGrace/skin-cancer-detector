from waitress import serve
from flask import Flask, request, jsonify, render_template, redirect
from flask import flash
from werkzeug.utils import secure_filename
from main import Predict
import random
import os
import json

#Save images to the 'static' folder as Flask serves images from this directory
UPLOAD_FOLDER = 'static/images/'

app = Flask(__name__, static_folder="static")

#Add reference fingerprint. 
#Cookies travel with a signature that they claim to be legit. 
#Legitimacy here means that the signature was issued by the owner of the cookie.
#Others cannot change this cookie as it needs the secret key. 
#It's used as the key to encrypt the session - which can be stored in a cookie.
#Cookies should be encrypted if they contain potentially sensitive information.
app.secret_key = "secret key"

#Define the upload folder to save images uploaded by the user. 
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def home_page():

    return render_template('index.html')

#Add Post method to the decorator to allow for form submission. 
@app.route('/', methods=['POST'])
def submit_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No file selected for uploading')
            return redirect(request.url)
        if file:
            filename = secure_filename(file.filename)  #Use this werkzeug method to secure filename. 
            file.save(os.path.join(app.config['UPLOAD_FOLDER'],filename))
            #getPrediction(filename)
            label = Predict(filename)
            flash(label)
            full_filename = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            flash(full_filename)
            return redirect('/')

if __name__=='__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
    #serve(app, host='0.0.0.0', port=port, debug=True)
 
