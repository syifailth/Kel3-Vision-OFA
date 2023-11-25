import base64, json
from flask import Flask, render_template, request, Response, session
from werkzeug.utils import secure_filename
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import plotly
import plotly.express as px
import plotly.graph_objects as go 
import cv2, io, math, os
from ultralytics import YOLO
import uuid
secret_key = uuid.uuid4().hex
print(secret_key)


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/files'
app.secret_key = '70c6d037a8ba454595ed72e7abf29379'

global metal_cls_fig, img_uploaded, metal_cls_pred

model_klasifikasi = tf.keras.models.load_model('model\klasifikasi-best.h5')
kelas_klasifikasi = ['Cacat_Korosi', 'Cacat_Las', 'Cacat_Lubang', 'Cacat_Retak']


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/prediction-classification', methods=['GET','POST'])
def prediction_classification():
    global metal_cls_fig, img_uploaded, kelas_klasifikasi, metal_cls_pred
    if request.method=='POST':
        file = request.files['file']

        img_bytes = file.read()
        img = tf.image.decode_jpeg(img_bytes, channels=3)
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = tf.image.resize(img, [224, 224])
        img = np.expand_dims(img, axis=0)
    
        metal_cls_pred = model_klasifikasi.predict(img)[0]
        metal_cls_pred = [round(p*100, 2) for p in metal_cls_pred]

        custom_colors = ['#9FE2BF', '#6495ED', '#DE3163', '#40E0D0']

        metal_cls_fig = go.Figure()
        metal_cls_fig.add_trace(go.Bar(
            y=kelas_klasifikasi,  
            x=metal_cls_pred,  
            orientation='h',
            marker_color=custom_colors[:len(kelas_klasifikasi)]
        ))

        metal_cls_fig.update_layout(
            xaxis_title='Probability',  
            yaxis_title='Class',  
            barmode='group',
            margin=dict(l=0, r=0, b=0, t=0),
            font=dict(size=14)
        )
        metal_cls_fig.update_layout(title_font=dict(color='black'))
        metal_cls_fig.update_layout(showlegend=False)
        metal_cls_fig.update_layout(template='plotly_white')
        metal_cls_fig.update_layout({
            'plot_bgcolor': 'rgba(0,0,0,0)',
            'paper_bgcolor': 'rgba(0,0,0,0)'
        })
        metal_cls_fig.update_xaxes(color='black', showgrid=False) 
        metal_cls_fig.update_yaxes(color='black', showgrid=False) 
        metal_cls_fig.update_layout(width=500, height=400)
        
        metal_cls_fig = json.dumps(metal_cls_fig, cls=plotly.utils.PlotlyJSONEncoder)
        
        img_uploaded = base64.b64encode(img_bytes).decode('utf-8')

    return render_template('prediksi-klasifikasi.html', metal_cls_chart=metal_cls_fig, img_uploaded=img_uploaded)



def detect_objects_on_image(image):
    model = YOLO("model/deteksi-best.pt")
    results = model.predict(image)
    result = results[0]

    output = []
    for box in result.boxes:
        x1, y1, x2, y2 = [round(x) for x in box.xyxy[0].tolist()]
        class_id = box.cls[0].item()
        prob = round(box.conf[0].item(), 2)
        output.append([x1, y1, x2, y2, result.names[class_id], prob])

    draw = ImageDraw.Draw(image)

    box_color = "blue"  

    for box in output:
        x1, y1, x2, y2, obj_type, prob = box
        text_x = x1 + 5  
        text_y = y1 + 5 

        draw.rectangle([x1, y1, x2, y2], outline=box_color, width=5)
        dynamic_font_size = min(30, int((x2 - x1) / 10))
        font = ImageFont.truetype("static/fonts/arialbd.ttf", dynamic_font_size)  
        draw.text((text_x, text_y), f"{obj_type} {prob}", fill=box_color, font=font)

    img_byte_array = io.BytesIO()
    image.save(img_byte_array, format='JPEG')
    img_byte_array.seek(0)
    encoded_image = base64.b64encode(img_byte_array.getvalue()).decode('utf-8')
    return encoded_image


@app.route('/detection-image', methods=['GET','POST'])
def detection_image():
    global img_uploaded, detect_image
    # detect_image = None

    if request.method == 'POST':
        file = request.files['file']
        img_bytes = file.read()
        img_uploaded = base64.b64encode(img_bytes).decode('utf-8')
        if file:
            img = Image.open(file)
            detect_image = detect_objects_on_image(img)

    return render_template('deteksi-gambar.html', result_image=detect_image, uploaded_img=img_uploaded)


def video_detection(source):
    video_capture = source
    cap=cv2.VideoCapture(video_capture)
    frame_width=int(cap.get(3))
    frame_height=int(cap.get(4))
    model=YOLO("model/deteksi-best.pt")
    classNames = ["Bad-Metal", "Good-Metal"]
    while True:
        success, img = cap.read()
        results=model(img,stream=True)
        for r in results:
            boxes=r.boxes
            for box in boxes:
                x1,y1,x2,y2=box.xyxy[0]
                x1,y1,x2,y2=int(x1), int(y1), int(x2), int(y2)
                print(x1,y1,x2,y2)
                cv2.rectangle(img, (x1,y1), (x2,y2), (255,0,255),3)
                conf=math.ceil((box.conf[0]*100))/100
                cls=int(box.cls[0])
                class_name=classNames[cls]
                label=f'{class_name}{conf}'
                t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]
                c2 = x1 + t_size[0], y1 - t_size[1] - 3
                cv2.rectangle(img, (x1,y1), c2, [255,0,255], -1, cv2.LINE_AA)  # filled
                cv2.putText(img, label, (x1,y1-2),0, 1,[255,255,255], thickness=1,lineType=cv2.LINE_AA)
        yield img

cv2.destroyAllWindows()


def generate_frames_web(source):
    yolo_output = video_detection(source)
    for detection_ in yolo_output:
        ref,buffer=cv2.imencode('.jpg',detection_)

        frame=buffer.tobytes()
        yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame +b'\r\n')
        

@app.route("/webcam")
def webcam():
    return render_template("webcam.html")


@app.route('/video-webcam')
def webcam_prediction():
    return Response(generate_frames_web(source=0), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/detection-video', methods=['GET','POST'])
def detection_video():
    if request.method == 'POST':
        file = request.files['file']
        file.save(os.path.join(os.path.abspath(os.path.dirname(__file__)), app.config['UPLOAD_FOLDER'],
                               secure_filename(file.filename)))  
        session['video_path'] = os.path.join(os.path.abspath(os.path.dirname(__file__)), app.config['UPLOAD_FOLDER'],
                                             secure_filename(file.filename))

    return render_template('deteksi-video.html')


def generate_frames(source = ''):
    yolo_output = video_detection(source)
    for detection_ in yolo_output:
        ref,buffer=cv2.imencode('.jpg',detection_)

        frame=buffer.tobytes()
        yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame +b'\r\n')
        

@app.route('/video')
def video():
    return Response(generate_frames(source = session.get('video_path', None)),mimetype='multipart/x-mixed-replace; boundary=frame')


                           
if __name__ == "__main__":
    app.run(debug=True) 