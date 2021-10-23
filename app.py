from flask import Flask, render_template, request,jsonify,render_template_string,url_for
from models import MobileNet
import os
from math import floor
import io
import traceback,sys,os
from service_streamer import ThreadedStreamer
from PIL import Image
import base64,uuid, datetime,glob
import pandas as pd
from pathlib import Path


app = Flask(__name__)

app.config['STATIC'] = "./static/"
app.config['UPLOAD_FOLDER'] = os.path.join(app.config['STATIC'],'uploads/')    # To host static images
app.config['datalog'] = os.path.join(app.config['STATIC'], 'data/data.csv')
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # Limit the memory to 10 MB.


columns = ["Inference Id","Prediction","Confidence","Timestamp"]

infer_baseblock = ("{% extends 'base.html' %}"+
"{% block content %}"+
"<div class=\"pricing-header px-3 py-3 pt-md-5 pb-md-4 mx-auto text-center\">"+
"<p class=\"lead\">File(s) uploaded successfully. You have uploaded an image of</p>")

infer = (
" <h3 class=\"display-4\">{{val1}}!</h3> <p class=\"mt-5\">Confidence: <strong>{{val2}}%</strong></p> <p class=\"mt-5\">" +
"<img src=\"data:image/jpeg;base64,{{ img_data }}\">"
)

# <img class="mb-4" src="data:image/jpeg;base64,{{ img_data }}" alt="" width="300">
# <img src="data:image/jpeg;base64,{{ img_data }}" alt="">


infer_endblock = ("<a href=\"\\\">Try again</a></p>"+
 "<h5 class=\"center\">Last 5 Inference Results</h5>"+
"<//div>" +
 "<div  class=\"center\">{{last5}}<//div>"+
"{% endblock %}")

model = MobileNet()
saveLocation = ""
streamer = ThreadedStreamer(model.infer, batch_size=64)



@app.route('/')
def index():
    df = pd.read_csv('./static/data/data.csv')
    df1 = df.tail(5)
    index_html = Path('./templates/index.html').read_text()    
    return render_template_string(index_html.replace('{{last_5_infer}}',df1.to_html(justify='center',index=False)).replace("<table","<table align=\"center\""))


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/infer', methods=['POST'])
def success():
    try:
        if request.method == 'POST':
            # print(static_url_path)
            
            saveLocation = []
            # filedata = []
            if 'file' in request.files:
                for f in request.files.getlist('file'):
                    # filedata.append(f)
                    print(f.filename)  
                    saveLocation.append(f.filename)
                    # saveLocation = f.filename
                    f.save(f.filename)
                
                return process(saveLocation)
    except :
            return jsonify({'error': traceback.print_exception(*sys.exc_info())})

def process(saveLocation):
    df = pd.read_csv('./static/data/data.csv')
    results = streamer.predict(saveLocation)
    print(results)

    output = ""
    count = 0
    pred_id = str(uuid.uuid4())[:8]

    if len(results) == 1:
        print("single")    
        im = Image.open(saveLocation[0])
        data = io.BytesIO()
        im.save(data, 'JPEG')
        encoded_img_data = base64.b64encode(data.getvalue())            
        os.remove(saveLocation[0])

        df = df.append({'Inference Id':pred_id, 'Prediction':results[0][0], 'Confidence':results[0][1],'Timestamp':datetime.datetime.now().strftime("%c")},ignore_index=True)
        df1 = df.tail(5)
        print(df1)
        df1.to_csv(app.config['datalog'], mode='a',  header=False, index=False)
        output = render_template_string(infer_baseblock+infer+infer_endblock.replace("{{last5}}",df1.to_html(justify='center',index=False)).replace("<table","<table align=\"center\""),val1=results[0][0],val2=results[0][1],img_data=encoded_img_data)                    
    else:                    
        for result in results:
            
            im = Image.open(saveLocation[count])
            data = io.BytesIO()
            im.save(data, 'JPEG')
            encoded_img_data = base64.b64encode(data.getvalue())            
            os.remove(saveLocation[count])

            output += infer.replace("{{val1}}",result[0]).replace("{{val2}}",result[1]) #.replace("{{ img_data }}",encoded_img_data)
            df = df.append({'Inference Id':pred_id, 'Prediction':result[0], 'Confidence':result[1],'Timestamp':datetime.datetime.now().strftime("%c")},ignore_index=True)
            count = count + 1

        df1 = df.tail(5)
        print(df1)
        df1.to_csv(app.config['datalog'], mode='a', header=False, index=False)
        
        output = render_template_string(infer_baseblock+output+infer_endblock.replace("{{last5}}",df1.to_html(justify='center',index=False)).replace("<table","<table align=\"center\""))                

    return output


if __name__ == '__main__':
    app.debug = True
    port = int(os.environ.get("PORT", 5000))    
    # app.config['sample_img'] = sys.argv[1]
    # print(app.config['sample_img'])
    # process(app.config['sample_img'])
    app.run(host='0.0.0.0', port=port, debug=True)
