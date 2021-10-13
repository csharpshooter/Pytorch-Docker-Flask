from flask import Flask, render_template, request,jsonify,render_template_string
from models import MobileNet
import os
from math import floor
import io
import traceback,sys,os
from service_streamer import ThreadedStreamer
import re

infer = ("{% extends \'base.html' %}"+
"{% block content %}"+
"<p class=\"lead\">File(s) uploaded successfully." +
"You have uploaded an image of</p> <h1 class=\"display-4\">{{val1}}!</h1> <p class=\"mt-5\">Confidence: <strong>{{val2}}%</strong></p> <p class=\"mt-5\">" +
"<a href=\"\\\">Try again</a></p>"+
"{% endblock %}")

app = Flask(__name__)
model = MobileNet()
saveLocation = ""
app.config['UPLOAD_FOLDER'] = 'uploads'
streamer = ThreadedStreamer(model.infer, batch_size=64)
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/infer', methods=['POST'])
def success():
    try:
        if request.method == 'POST':
            saveLocation = []
            if 'file' in request.files:
                for f in request.files.getlist('file'):
                    print(f.filename)  
                    saveLocation.append(f.filename)
                    # saveLocation = f.filename
                    f.save(f.filename)
                result = streamer.predict(saveLocation)
                # result = model.infer(saveLocation)
                print(result)

                if len(result) == 1:
                    print("single")
                    
                    return render_template_string(infer,val1=result[0][0],val2=result[0][1])
                else:
                    print("multiple")
                
    except :
            return jsonify({'error': traceback.print_exception(*sys.exc_info())})

if __name__ == '__main__':
    app.debug = True
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
