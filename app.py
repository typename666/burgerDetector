from flask import Flask, request, render_template, jsonify
import cv2
import numpy as np
from base64 import b64encode
import time

import onnxruntime as rt

IMG_SIZE= (150, 150)
THRESHOLD= 0.7

session = rt.InferenceSession('model.onnx')
inputDetails = session.get_inputs()

def detectBurger(img):
    img= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)/255.0

    roi_size= min(img.shape[:2])//2

    x1, y1, x2, y2= 0, 0, img.shape[1], img.shape[0]
    while roi_size>=min(img.shape[:2])*0.05:
        xMin, xMax, yMin, yMax= x1, x2, y1, y2
        step= roi_size//4
        xRange, yRange= range(xMin, xMax-roi_size, step), range(yMin, yMax-roi_size, step)

        rois= np.zeros((len(xRange)*len(yRange), *IMG_SIZE))
        boxes= np.zeros((rois.shape[0], 4), dtype= np.int32)

        i= 0
        for x in xRange:
            for y in yRange:
                x1, y1, x2, y2= x, y, x+roi_size, y+roi_size
                roi= cv2.resize(img[y1:y2, x1:x2], IMG_SIZE)

                rois[i]= roi
                boxes[i]= x1, y1, x2, y2
                i+= 1

        roi_size//= 2

        rois= np.array(rois, dtype= np.float32)
        rois= rois.reshape(*rois.shape, 1)

        pred= np.array(session.run(None, {inputDetails[0].name: rois}))[0]

        maxInd= int(np.argmax(pred, 0))

        x1, y1, x2, y2= boxes[maxInd]
        if pred[maxInd]<THRESHOLD:
            return x1, y1, x2, y2

    return x1, y1, x2, y2

def rescaleImg(img, scale, box):
   newH, newW= int(img.shape[0]*scale), int(img.shape[1]*scale)
   box= int(box[0]*scale), int(box[1]*scale), int(box[2]*scale), int(box[3]*scale)

   return cv2.resize(img, (newW, newH)), box

app = Flask(__name__)

@app.route('/')
def index():
  return render_template("index.html")

@app.route('/api', methods=["POST"])
def api():
    file = request.files["img"]

    filestr = file.read()
    npimg = np.fromstring(filestr, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    start= time.time()
    box= detectBurger(img)
    wallTime= time.time()-start

    if box!= (0, img.shape[0], 0, img.shape[1]):
        img, box= rescaleImg(img, 0.3, box)
        img= cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)

    _, img_encoded = cv2.imencode('.jpg', img)

    b64_mystring = b64encode(img_encoded).decode("utf-8")

    return jsonify({ "img":  str(b64_mystring), 'time': '{:.2f} с'.format(wallTime)})

if __name__ == '__main__':
  app.run()