from flask import Flask, render_template, request,jsonify
from models import MobileNet
import os
from math import floor
import io
import traceback,sys,os


app = Flask(__name__)
model = MobileNet()
saveLocation = ""
app.config['UPLOAD_FOLDER'] = 'uploads'

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
                # result = streamer.predict(saveLocation)
                result = model.infer(saveLocation)
                print(result)
                # inference, confidence = result.__getitem__(0).split('|')
                # return render_template('inference.html', name= result.__getitem__(0), confidence= result.__getitem__(1))
                return render_template('inference.html', name= "", confidence= result)
    except :
            return jsonify({'error': traceback.print_exception(*sys.exc_info())})

if __name__ == '__main__':
    app.debug = True
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
