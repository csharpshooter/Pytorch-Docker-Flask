from flask import Flask, render_template, request,jsonify,render_template_string
from models import MobileNet
import os
from math import floor
import io
import traceback,sys,os
from service_streamer import ThreadedStreamer
import re

infer_baseblock = ("{% extends 'base.html' %}"+
"{% block content %}"+
"<div class=\"pricing-header px-3 py-3 pt-md-5 pb-md-4 mx-auto text-center\">"+
"<p class=\"lead\">File(s) uploaded successfully. You have uploaded an image of</p>")

infer = (
" <h3 class=\"display-4\">{{val1}}!</h3> <p class=\"mt-5\">Confidence: <strong>{{val2}}%</strong></p> <p class=\"mt-5\">" +
"<img class=\"mb-4\" src=\"data:image/jpeg;base64,{{ img_data }}\" alt=\"\" width=\"300\">"
)

infer_endblock = ("<a href=\"\\\">Try again</a></p>"+
"<//div>" +
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
                results = streamer.predict(saveLocation)
                print(results)

                output = ""

                if len(results) == 1:
                    print("single")                    
                    output = render_template_string(infer_baseblock+infer+infer_endblock,val1=results[0][0],val2=results[0][1])
                else:                    
                    for result in results:
                        output += infer.replace("{{val1}}",result[0]).replace("{{val2}}",result[1])
                    output = render_template_string(infer_baseblock+output+infer_endblock)

                return output             
    except :
            return jsonify({'error': traceback.print_exception(*sys.exc_info())})

if __name__ == '__main__':
    app.debug = True
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
