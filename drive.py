import argparse
import base64
from datetime import datetime
import os
import shutil

import numpy as np
import socketio
import eventlet
import eventlet.wsgi
from PIL import Image
from flask import Flask
from io import BytesIO

import torch
import torch.nn as nn
from torchvision.transforms import v2, ToPILImage
from src.model import CNN

sio = socketio.Server()
app = Flask(__name__)
model = None
prev_image_array = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MAX_SPEED = 30
MIN_SPEED = 10
speed_limit = MAX_SPEED

# Preprocessing image
transform = v2.Compose([
    v2.ToImage(),
    # v2.Resize((66, 200), antialias=True),
    v2.ToDtype(torch.float32, scale=True),  # Normalize expects float input
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

@sio.on('telemetry')
def telemetry(sid, data):
    if data:
        # The current steering angle of the car
        steering_angle = float(data["steering_angle"])
        # The current throttle of the car
        throttle = float(data["throttle"])
        # The current speed of the car
        speed = float(data["speed"])
        # The current image from the center camera of the car
        imgString = data["image"]
        image = Image.open(BytesIO(base64.b64decode(imgString)))
        image = transform(image).unsqueeze(0).to(device)
        # Predict
        model.eval()
        with torch.inference_mode():
            steering_angle = model(image).item()

        # Adjust speed
        global speed_limit
        if speed > speed_limit:
            speed_limit = MIN_SPEED 
        else:
            speed_limit = MAX_SPEED

        throttle = 1.0 - steering_angle**2 - (speed/speed_limit)**2

        print('{} {} {}'.format(steering_angle, throttle, speed))
        send_control(steering_angle, throttle)

        # save frame
        if args.image_folder != '':
            timestamp = datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S_%f')[:-3]
            image_filename = os.path.join(args.image_folder, timestamp)
            image = image[0]
            pil_image = ToPILImage()(image)
            pil_image.save('{}.jpg'.format(image_filename))
    else:
        # NOTE: DON'T EDIT THIS.
        sio.emit('manual', data={}, skip_sid=True)


@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit(
        "steer",
        data={
            'steering_angle': steering_angle.__str__(),
            'throttle': throttle.__str__()
        },
        skip_sid=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument(
        'model',
        type=str,
        help='Path to model .pth file. Model should be on the same path.'
    )
    parser.add_argument(
        'image_folder',
        type=str,
        nargs='?',
        default='',
        help='Path to image folder. This is where the images from the run will be saved.'
    )
    args = parser.parse_args()

    # Load the PyTorch model
    model = CNN()
    model.load_state_dict(torch.load(args.model, map_location=device, weights_only=True))
    model.to(device)

    if args.image_folder != '':
        print("Creating image folder at {}".format(args.image_folder))
        if not os.path.exists(args.image_folder):
            os.makedirs(args.image_folder)
        else:
            shutil.rmtree(args.image_folder)
            os.makedirs(args.image_folder)
        print("RECORDING THIS RUN ...")
    else:
        print("NOT RECORDING THIS RUN ...")

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)