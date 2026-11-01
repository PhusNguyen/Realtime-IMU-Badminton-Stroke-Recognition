"""
Real-time badminton stroke recognition with the trained CNN.

Workflow
    1. Listens for UDP packets from the racket sensor on port 44444. Each
       240-byte packet holds 10 samples of six big-endian float32 values:
       AccX, AccY, AccZ, GyX, GyY, GyZ.
    2. Plots the incoming signals live. Perform strokes, then close the plot
       window to end the session.
    3. Saves the session, segments it into strokes, and prints the stroke
       count and the predicted class (1-8) for each stroke.

Requires
    DeepLearning/models/dl_optimize.pth
        Trained PyTorch weights for StrokeClassifier (defined below). The
        class must match the one used in training for the weights to load.
"""
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import socket
import struct
import json
import numpy as np
from scipy.ndimage import gaussian_filter1d
import torch
import torch.nn as nn
import os
import shutil

# ---------------------------------------------------------------------------
# Configuration: all paths are relative to this script's folder
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Working folders for the current session; both are cleared on every run
RECORDING_DIR = os.path.join(BASE_DIR, 'newdata_prediction', 'quick_large_file')
SEGMENT_DIR = os.path.join(BASE_DIR, 'newdata_prediction', 'quick_testing_newdata')
# Trained CNN weights
MODEL_PATH = os.path.join(BASE_DIR, 'DeepLearning', 'models', 'dl_optimize.pth')


# ---------------------------------------------------------------------------
# Model definition: must match the class used to train dl_optimize.pth
# ---------------------------------------------------------------------------
class StrokeClassifier(nn.Module):
    """CNN for 8-class stroke classification (805,592 parameters).

    Input:  (batch, 1, 100, 6)  one 100-sample window of 6 IMU channels
    Output: (batch, 8)          class scores
    """

    def __init__(self, num_classes=8, dropout=0.3):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=2)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(32 * 98 * 4, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.dropout(self.bn1(torch.relu(self.conv1(x))))
        x = self.dropout(torch.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


def load_dl_model(path):
    """Load dl_optimize.pth, whether it holds a state_dict or a full model."""
    try:
        obj = torch.load(path, map_location='cpu', weights_only=True)
    except Exception:
        # the file stores a full pickled model rather than only weights
        obj = torch.load(path, map_location='cpu', weights_only=False)

    if isinstance(obj, nn.Module):
        model = obj
    else:
        if isinstance(obj, dict) and 'model_state_dict' in obj:
            obj = obj['model_state_dict']
        model = StrokeClassifier()
        # strict loading fails loudly if the class above doesn't match training
        model.load_state_dict(obj)

    model.eval()
    return model

################################# store uncut data ##################################
# bind all IP
HOST = '0.0.0.0'
# Listen on Port
PORT = 44444
#Size of receive buffer
BUFFER_SIZE = 1024
# Create a UDP socket
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
# Bind the socket to the host and port
s.bind((HOST, PORT))

# create fixed arrays and variables for plotting
time_arr = [None]*1000
AccX_arr = [None]*1000
AccY_arr = [None]*1000
AccZ_arr = [None]*1000
Gyx_arr = [None]*1000
Gyy_arr = [None]*1000
Gyz_arr = [None]*1000
t = 0
b = 0

# create dictionary including array for storing data
data = {
    "AccX" : [],
    "AccY" : [],
    "AccZ" : [],
    "GyX"  : [],
    "GyY"  : [],
    "GyZ"  : [],
    "frame": []
}

# set up for plotting figure
plt.style.use('dark_background')
fig = plt.figure(figsize=(10,5))
ln1, = plt.plot(time_arr, AccX_arr)
ln2, = plt.plot(time_arr, AccY_arr)
ln3, = plt.plot(time_arr, AccZ_arr)
ln4, = plt.plot(time_arr, Gyx_arr)
ln5, = plt.plot(time_arr, Gyy_arr)
ln6, = plt.plot(time_arr, Gyz_arr)
plt.ylim(-180,180)

fig.legend([ln1, ln2, ln3, ln4, ln5, ln6],['AcX','AcY','AcZ','GyX','GyY','GyZ'])
fig.tight_layout()

# function to update data per frame
def animate(AccX):
    global t,b,i
    global time_arr, AccX_arr, AccY_arr, AccZ_arr, Gyx_arr, Gyy_arr, Gyz_arr

    # receive message
    message, address = s.recvfrom(BUFFER_SIZE)

    # convert the raw bytes to a hex string
    message = message.hex()

    for i in range(10): # range is the quantity X of packet (X = 6 data)
        # split an array of bytes into format: 8-bit hexadecimal
        AccX = message[b:b+8]; b += 8
        AccY = message[b:b+8]; b += 8
        AccZ = message[b:b+8]; b += 8
        Gyx = message[b:b+8]; b += 8
        Gyy = message[b:b+8]; b += 8
        Gyz = message[b:b+8]; b += 8
        # max is 480 for 240-byte packet

        # convert hex to float
        AccX = struct.unpack('!f', bytes.fromhex(AccX))[0]
        AccY = struct.unpack('!f', bytes.fromhex(AccY))[0]
        AccZ = struct.unpack('!f', bytes.fromhex(AccZ))[0]
        Gyx = struct.unpack('!f', bytes.fromhex(Gyx))[0]
        Gyy = struct.unpack('!f', bytes.fromhex(Gyy))[0]
        Gyz = struct.unpack('!f', bytes.fromhex(Gyz))[0]

        # append to json file
        data['AccX'] += [AccX]
        data['AccY'] += [AccY]
        data['AccZ'] += [AccZ]
        data['GyX']  += [Gyx]
        data['GyY']  += [Gyy]
        data['GyZ']  += [Gyz]
        data['frame']+= [t]

        # append to plot realtime
        time_arr.pop(0); time_arr.append(t); t += 1
        AccX_arr.pop(0); AccX_arr.append(AccX)
        AccY_arr.pop(0); AccY_arr.append(AccY)
        AccZ_arr.pop(0); AccZ_arr.append(AccZ)
        Gyx_arr.pop(0); Gyx_arr.append(Gyx)
        Gyy_arr.pop(0); Gyy_arr.append(Gyy)
        Gyz_arr.pop(0); Gyz_arr.append(Gyz)

        ln1.set_data(time_arr, AccX_arr)
        ln2.set_data(time_arr, AccY_arr)
        ln3.set_data(time_arr, AccZ_arr)
        ln4.set_data(time_arr, Gyx_arr)
        ln5.set_data(time_arr, Gyy_arr)
        ln6.set_data(time_arr, Gyz_arr)

    b = 0 # reset full cycle of a message (48*10)

    fig.gca().relim()
    fig.gca().autoscale_view()

    return ln1, ln2, ln3, ln4, ln5, ln6,

animation = FuncAnimation(fig, animate ,interval=90, cache_frame_data=False)
plt.show()

# refresh directory for new prediction
file1 = RECORDING_DIR
file2 = SEGMENT_DIR
# clear the working folders (create them if they don't exist yet)
for folder in (file1, file2):
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)

path = RECORDING_DIR + os.sep
path2 = SEGMENT_DIR + os.sep
file_name = "text"
with open(path + file_name + '.json', 'w') as f:
    json.dump(data, f, indent=4)

s.close()

################################### segmentation ################################
# Open Json file
with open(path + file_name + '.json', 'r') as file:
    data = json.load(file)

# Access the data
AccX = data["AccX"]
AccY = data["AccY"]
AccZ = data["AccZ"]
GyX =  data["GyX"]
GyY =  data["GyY"]
GyZ =  data["GyZ"]
frame= data["frame"]

# create dictionary including array for storing data
data = {
    "AccX" : [],
    "AccY" : [],
    "AccZ" : [],
    "GyX"  : [],
    "GyY"  : [],
    "GyZ"  : [],
    "frame": []
}

# create array for accelerometer and gyroscope
Acc = []
Gy = []

# threshold to store
Acc_threshold = []
frame_threshold = []

# Calculate RMS value of Accelermeter and Gyroscope
for i in range(len(frame)):
    Acc.append(float(np.sqrt((AccX[i]*AccX[i] + AccY[i]*AccY[i] + AccZ[i]*AccZ[i])/3)))

# apply a Gaussian lowpass filter
filtered = gaussian_filter1d(Acc, sigma=5)

# plot the original data next to the filtered data
# plt.figure(figsize=(10, 4))

# plt.subplot(121)
# plt.plot(frame, Acc)
# plt.title("Signal with Noise")
# plt.margins(0, .05)

# plt.subplot(122)
# plt.plot(frame, filtered)
# plt.title("Filtered Signal")
# plt.margins(0, .05)

# plt.tight_layout()
# plt.show()

# Shot indicator
# threshold_value = int(input("Enter threshold: "))
for i in range(len(filtered)):
    if filtered[i] >= 35: # threshold to evaluate, different for each stroke type
        Acc_threshold.append(Acc[i])
        frame_threshold.append(frame[i])
    else:
        Acc_threshold.append(None)
        frame_threshold.append(None)

count = 0 # Shot count
y1 = []; y2 = []
# count based on the datatype change (from None to Float)
for i in range(1,len(Acc_threshold),1):
    if type(Acc_threshold[i-1]) != float and type(Acc_threshold[i]) == float:
        count += 1
        y1.append(frame_threshold[i])     # index of the fist float value
    elif type(Acc_threshold[i-1]) == float and type(Acc_threshold[i]) != float:
        y2.append(frame_threshold[i-1])   # index of the last float value

Acc_max = []
frame_max = []
Acc_max_value = 0

# find the max value and its frame as marker or middle value of the signal
for i in range(len(y1)):
    middle_index = int(np.round(np.abs(y1[i]-y2[i])/2))
    frame_max.append( y1[i] + middle_index )

### if two peaks is too close (less than 50 frame), the second one will be eliminated 
for i in range(len(frame_max)-1):
    try:
        if frame_max[i+1] - frame_max[i] < 50:
            frame_max.pop(i+1)
    except:
        break

# print(f"shot count after process: {len(frame_max)}")
# print(f"frame of peak value after process: {frame_max}")

### From the peak frame, stroke data are cut
for i in range(len(frame_max)):
    for j in range(frame_max[i] - 100 , frame_max[i] + 100):
        # append to json file
        data['AccX'] += [AccX[j]]
        data['AccY'] += [AccY[j]]
        data['AccZ'] += [AccZ[j]]
        data['GyX']  += [GyX[j]]
        data['GyY']  += [GyY[j]]
        data['GyZ']  += [GyZ[j]]
        data['frame']+= [frame[j]]

    with open(path2 + str(i) + '.json', 'w') as f:
        json.dump(data, f, indent=4)

    data = {
    "AccX" : [],
    "AccY" : [],
    "AccZ" : [],
    "GyX"  : [],
    "GyY"  : [],
    "GyZ"  : [],
    "frame": []
    }

X = []
### input info
for i in range(count):
    file_name = f"{path2}{str(i)}"
    with open(  file_name + ".json", 'r') as file:
        data = json.load(file)

    # Access the data from the file
    AccX = data["AccX"]
    AccY = data["AccY"]
    AccZ = data["AccZ"]
    GyX =  data["GyX"]
    GyY =  data["GyY"]
    GyZ =  data["GyZ"]

    # append original data of 8 classes and its label
    X.append([AccX[50:150], AccY[50:150], AccZ[50:150], GyX[50:150], GyY[50:150], GyZ[50:150]])

# load trained model from pth file
model = load_dl_model(MODEL_PATH)

# resize input to (strokes, channel, time, axes) for PyTorch; keeps the same
# element order as the original Keras reshape to (strokes, 100, 6, 1)
X = np.array(X, dtype=np.float32)
X = X.reshape(X.shape[0], 1, 100, 6)

# print shot count
print("Shot count: ", count)

# evaluate model
with torch.no_grad():
    y_predict = model(torch.from_numpy(X)).numpy()
y_pred_classes = np.argmax(y_predict, axis=1)
print(y_pred_classes+1)