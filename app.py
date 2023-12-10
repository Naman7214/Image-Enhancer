from flask import Flask, request, render_template
import os.path as osp
import glob
import cv2
import numpy as np
import torch
from RRDBNet_arch import RRDBNet

app = Flask(__name__)

model_path = 'models/RRDB_ESRGAN_x4.pth'
device = torch.device('cpu')

test_img_folder = 'LR/*'

model = RRDBNet(3, 3, 64, 23, gc=32)
model.load_state_dict(torch.load(model_path, map_location=device), strict=True)
model.eval()
model = model.to(device)

print('Model path {:s}. \nFlask app is running...'.format(model_path))

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/process_image', methods=['POST'])
def process_image():
    # Get file from the POST request
    uploaded_file = request.files['image']
    
    if uploaded_file.filename != '':
        # Save the uploaded image to a temporary file
        temp_path = 'temp.jpg'
        uploaded_file.save(temp_path)

        # Read and process the image
        img = cv2.imread(temp_path, cv2.IMREAD_COLOR)
        img = img * 1.0 / 255
        img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
        img_LR = img.unsqueeze(0)
        img_LR = img_LR.to(device)

        with torch.no_grad():
            output = model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()
        output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
        output = (output * 255.0).round()

        # Save the processed image
        result_path = 'static/result.png'
        cv2.imwrite(result_path, output)

        return render_template('result.html', result_path=result_path)

if __name__ == '__main__':
    app.run(debug=True)