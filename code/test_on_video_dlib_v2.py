# The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
#
#   This example shows how to run a CNN based face detector using dlib.  The
#   example loads a pretrained model and uses it to find faces in images.  The
#   CNN model is much more accurate than the HOG based model shown in the
#   face_detector.py example, but takes much more computational power to
#   run, and is meant to be executed on a GPU to attain reasonable speed.
#
#   You can download the pre-trained model from:
#       http://dlib.net/files/mmod_human_face_detector.dat.bz2
#
#   The examples/faces folder contains some jpg images of people.  You can run
#   this program on them and see the detections by executing the
#   following command:
#       ./cnn_face_detector.py mmod_human_face_detector.dat ../examples/faces/*.jpg
#
#
# COMPILING/INSTALLING THE DLIB PYTHON INTERFACE
#   You can install dlib using the command:
#       pip install dlib
#
#   Alternatively, if you want to compile dlib yourself then go into the dlib
#   root folder and run:
#       python setup.py install
#   or
#       python setup.py install --yes USE_AVX_INSTRUCTIONS --yes DLIB_USE_CUDA
#   if you have a CPU that supports AVX instructions, you have an Nvidia GPU
#   and you have CUDA installed since this makes things run *much* faster.
#
#   Compiling dlib should work on any operating system so long as you have
#   CMake and boost-python installed.  On Ubuntu, this can be done easily by
#   running the command:
#       sudo apt-get install libboost-python-dev cmake
#
#   Also note that this example requires scikit-image which can be installed
#   via the command:
#       pip install scikit-image
#   Or downloaded from http://scikit-image.org/download.html.

import sys
import dlib
import cv2

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.backends.cudnn as cudnn
import torchvision
import torch.nn.functional as F

from skimage import io
from PIL import Image

import datasets, hopenet, utils

if len(sys.argv) < 3:
    print(
        "Call this program like this:\n"
        "   ./cnn_face_detector.py mmod_human_face_detector.dat ../examples/faces/*.jpg\n"
        "You can get the mmod_human_face_detector.dat file from:\n"
        "    http://dlib.net/files/mmod_human_face_detector.dat.bz2")
    exit()

cnn_face_detector = dlib.cnn_face_detection_model_v1(sys.argv[1])

# ResNet50 structure
model = hopenet.Hopenet(torchvision.models.resnet.Bottleneck,
                        [3, 4, 6, 3], 66)

print 'Loading snapshot.'
# Load snapshot
saved_state_dict = torch.load("../snapshots/hopenet_alpha1.pkl")
model.load_state_dict(saved_state_dict)

print 'Loading data.'

transformations = transforms.Compose([transforms.Scale(224),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize(
                                          mean=[0.485, 0.456, 0.406],
                                          std=[0.229, 0.224, 0.225])])

gpu = 0
model.cuda(gpu)

# Test the Model
model.eval()  # Change model to 'eval' mode (BN uses moving mean/var).
total = 0

idx_tensor = [idx for idx in xrange(66)]
idx_tensor = torch.FloatTensor(idx_tensor).cuda(gpu)

win = dlib.image_window()

video = cv2.VideoCapture(sys.argv[2])
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))  # float
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))  # float

fps = int(video.get(cv2.CAP_PROP_FPS))
output_string = "test"

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out = cv2.VideoWriter('output/video/output-%s.avi' % output_string,
                      fourcc, fps, (width, height))

txt_out = open('output/video/output-%s.txt' % output_string, 'w')

frame_num = 1
n_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

while frame_num <= n_frames:
    print frame_num

    ret, frame = video.read()
    if ret is False:
        break

    cv2_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Dlib detect
    dets = cnn_face_detector(cv2_frame, 1)

    for idx, det in enumerate(dets):
        # Get x_min, y_min, x_max, y_max, conf
        x_min = det.rect.left()
        y_min = det.rect.top()
        x_max = det.rect.right()
        y_max = det.rect.bottom()
        conf = det.confidence

        if conf > 1.0:
            bbox_width = abs(x_max - x_min)
            bbox_height = abs(y_max - y_min)
            x_min -= 2 * bbox_width / 4
            x_max += 2 * bbox_width / 4
            y_min -= 3 * bbox_height / 4
            y_max += bbox_height / 4
            x_min = max(x_min, 0)
            y_min = max(y_min, 0)
            x_max = min(frame.shape[1], x_max)
            y_max = min(frame.shape[0], y_max)
            # Crop image
            img = cv2_frame[y_min:y_max, x_min:x_max]
            img = Image.fromarray(img)

            # Transform
            img = transformations(img)
            img_shape = img.size()
            img = img.view(1, img_shape[0], img_shape[1], img_shape[2])
            img = Variable(img).cuda(gpu)

            yaw, pitch, roll = model(img)

            yaw_predicted = F.softmax(yaw)
            pitch_predicted = F.softmax(pitch)
            roll_predicted = F.softmax(roll)
            # Get continuous predictions in degrees.
            yaw_predicted = torch.sum(
                yaw_predicted.data[0] * idx_tensor) * 3 - 99
            pitch_predicted = torch.sum(
                pitch_predicted.data[0] * idx_tensor) * 3 - 99
            roll_predicted = torch.sum(
                roll_predicted.data[0] * idx_tensor) * 3 - 99

            # Print new frame with cube and axis
            txt_out.write(str(frame_num) + '\t%f\t%f\t%f\n' % (
                yaw_predicted, pitch_predicted, roll_predicted))
            utils.plot_pose_cube(frame, yaw_predicted, pitch_predicted,
                                 roll_predicted, (x_min + x_max) / 2,
                                 (y_min + y_max) / 2, size=bbox_width)
            utils.draw_axis(frame, yaw_predicted, pitch_predicted,
                            roll_predicted, tdx=(x_min + x_max) / 2,
                            tdy=(y_min + y_max) / 2,
                            size=bbox_height / 2)
            # Plot expanded bounding box
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max),
                          (0, 255, 0), 1)

    out.write(frame)
    frame_num += 1

out.release()
video.release()
