# -*- coding: utf-8 -*-
"""
Real-time badminton stroke recognition with the trained SVM.

Workflow
    1. Listens for UDP packets from the racket sensor on port 44444. Each
       240-byte packet holds 10 samples of six big-endian float32 values:
       AccX, AccY, AccZ, GyX, GyY, GyZ.
    2. Plots the incoming signals live. Perform strokes, then close the plot
       window to end the session.
    3. Saves the session, segments it into strokes, extracts the 122
       features, and prints the stroke count and the predicted class (1-8)
       for each stroke.

Requires
    ClassicalML/models/svm_model.pkl
        Trained scikit-learn model that takes the 122 raw features and
        handles standardization and PCA itself (e.g. a Pipeline).

Created on Fri Feb 16 15:11:01 2024
@author: Phu Nguyen
"""

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import socket
import struct
import json
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.stats import skew, kurtosis
from scipy.stats import iqr, entropy
import pickle
import os
import shutil

# ---------------------------------------------------------------------------
# Configuration: all paths are relative to this script's folder
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Working folders for the current session; both are cleared on every run
RECORDING_DIR = os.path.join(BASE_DIR, 'newdata_prediction', 'quick_large_file')
SEGMENT_DIR = os.path.join(BASE_DIR, 'newdata_prediction', 'quick_testing_newdata')
# Trained classifier (standardization + PCA + SVM)
SVM_MODEL_PATH = os.path.join(BASE_DIR, 'ClassicalML', 'models', 'svm_model.pkl')

def update(frame):
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



def calculate_rms(data):
    squared_values = np.square(data)
    mean_squared = np.mean(squared_values)
    rms = np.sqrt(mean_squared)
    return rms
# function to calculate mean absolute deviation
def calculate_mad(data):
    mean = np.mean(data)
    absolute_deviations = np.abs(data - mean)
    mad = np.mean(absolute_deviations)
    return mad
# function to calculate signal
def calculate_sma(data):
    sma = np.sum(np.abs(data)) / len(data)
    return sma
# function to calculate movement intensity
def calculate_mi(x,y,z):
    mi_arr = 0
    mi_sum = 0
    for i in range(len(x)):
        mi_arr = np.sqrt(x[i]*x[i] + y[i]*y[i] + z[i]*z[i])
        mi_sum = mi_sum + mi_arr
    return mi_sum

def main():
    global HOST, PORT, BUFFER_SIZE
    global s, time_arr, AccX_arr, AccY_arr, AccZ_arr, Gyx_arr, Gyy_arr, Gyz_arr
    global t, b, data, fig, ln1, ln2, ln3, ln4, ln5, ln6


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

    animation = FuncAnimation(fig, update ,interval=90, cache_frame_data=False)
    plt.show()

    # working folders for this session
    file1 = RECORDING_DIR
    file2 = SEGMENT_DIR
    # clear the working folders (create them if they don't exist yet)
    for folder in (file1, file2):
        if os.path.exists(folder):
            shutil.rmtree(folder)
        os.makedirs(folder)

    path = RECORDING_DIR + os.sep
    path2 = SEGMENT_DIR + os.sep

    file_name = "test"
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
        if filtered[i] >= 35: # threshold to evaluate
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

    # find the max value and its frame as marker or median value of the signal
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
    ################################### Extract feature ################################
    ### create feature array to store data
    # max
    max_Ax = []; max_Ay = []; max_Az = []
    max_Gx = []; max_Gy = []; max_Gz = []
    max_Acc = []; max_G = []

    # min
    min_Ax = []; min_Ay = []; min_Az = []
    min_Gx = []; min_Gy = []; min_Gz = []

    # mean
    mean_Ax = []; mean_Ay = []; mean_Az = []
    mean_Gx = []; mean_Gy = []; mean_Gz = []
    mean_Acc = []; mean_G = []

    # RMS
    rms_Ax = []; rms_Ay = []; rms_Az = []
    rms_Gx = []; rms_Gy = []; rms_Gz = []

    # Mean absolute deviation (mad)
    mad_Ax = []; mad_Ay = []; mad_Az = []
    mad_Gx = []; mad_Gy = []; mad_Gz = []
    mad_Acc = []; mad_G = []

    # median
    median_Ax = []; median_Ay = []; median_Az = []
    median_Gx = []; median_Gy = []; median_Gz = []
    median_Acc = []; median_G = []

    # std
    std_Ax = []; std_Ay = []; std_Az = []
    std_Gx = []; std_Gy = []; std_Gz = []
    std_Acc = []; std_G = []

    # variance
    var_Ax = []; var_Ay = []; var_Az = []
    var_Gx = []; var_Gy = []; var_Gz = []
    var_Acc = []; var_G = []

    # Covariance
    cov_AxAy = []; cov_AxAz = []; cov_AyAz = []; cov_AxGx = []; cov_AxGy = []; cov_AxGz = []
    cov_GxGy = []; cov_GxGz = []; cov_GyGz = []; cov_AyGx = []; cov_AyGy = []; cov_AyGz = []
    cov_AzGx = []; cov_AzGy = []; cov_AzGz = []

    # Correlation Coeficient
    cor_AxAy = []; cor_AxAz = []; cor_AyAz = []; cor_AxGx = []; cor_AxGy = []; cor_AxGz = []
    cor_GxGy = []; cor_GxGz = []; cor_GyGz = []; cor_AyGx = []; cor_AyGy = []; cor_AyGz = []
    cor_AzGx = []; cor_AzGy = []; cor_AzGz = []

    # Skewness
    skew_Ax = []; skew_Ay = []; skew_Az = []
    skew_Gx = []; skew_Gy = []; skew_Gz = []
    skew_Acc = []; skew_G = []

    # Kurtosis
    kur_Ax = []; kur_Ay = []; kur_Az = []
    kur_Gx = []; kur_Gy = []; kur_Gz = []
    kur_Acc = []; kur_G = []

    # Signal magnitude area (sma)
    sma_Ax = []; sma_Ay = []; sma_Az = []
    sma_Gx = []; sma_Gy = []; sma_Gz = []
    sma_Acc = []; sma_G = []

    # Movement Intensity (mi)
    mi_Acc = []; mi_G = []

    # Crest Factor (cf)
    cf_Ax = []; cf_Ay = []; cf_Az = []
    cf_Gx = []; cf_Gy = []; cf_Gz = []
    cf_Acc = []; cf_G = []

    # Interquartile Range (ir)
    ir_Ax = []; ir_Ay = []; ir_Az = []
    ir_Gx = []; ir_Gy = []; ir_Gz = []
    ir_Acc = []; ir_G = []

    # Energy
    energy_Ax = []; energy_Ay = []; energy_Az = []
    energy_Gx = []; energy_Gy = []; energy_Gz = []
    energy_Acc = []; energy_G = []

    # Entropy
    entropy_Ax = []; entropy_Ay = []; entropy_Az = []
    entropy_Gx = []; entropy_Gy = []; entropy_Gz = []
    entropy_Acc = []; entropy_G = []

    frame = []

    ### create json file to store data to json file
    features_data = {
        "max_Ax" : [],"max_Ay" : [],"max_Az" : [],
        "max_Gx"  : [],"max_Gy"  : [],"max_Gz"  : [],

        "min_Ax" : [], "min_Ay" : [], "min_Az" : [],
        "min_Gx"  : [], "min_Gy"  : [], "min_Gz"  : [],

        "mean_Ax" : [],"mean_Ay" : [],"mean_Az" : [],
        "mean_Gx"  : [],"mean_Gy"  : [],"mean_Gz"  : [],

        "rms_Ax" : [], "rms_Ay" : [], "rms_Az" : [],
        "rms_Gx"  : [], "rms_Gy"  : [], "rms_Gz"  : [],

        "mad_Ax" : [], "mad_Ay" : [], "mad_Az" : [],
        "mad_Gx"  : [], "mad_Gy"  : [], "mad_Gz"  : [],

        "median_Ax" : [],"median_Ay" : [],"median_Az" : [],
        "median_Gx"  : [],"median_Gy"  : [],"median_Gz"  : [],

        "std_Ax" : [],"std_Ay" : [],"std_Az" : [],
        "std_Gx"  : [],"std_Gy"  : [],"std_Gz"  : [],

        "var_Ax" : [],"var_Ay" : [],"var_Az" : [],
        "var_Gx"  : [],"var_Gy"  : [],"var_Gz"  : [],

        "cov_AxAy" : [], "cov_AxAz" : [], "cov_AyAz" : [], "cov_AxGx" : [], "cov_AxGy" : [], "cov_AxGz" : [],
        "cov_GxGy" : [], "cov_GxGz" : [], "cov_GyGz" : [], "cov_AyGx" : [], "cov_AyGy" : [], "cov_AyGz" : [],
        "cov_AzGx" : [], "cov_AzGy" : [], "cov_AzGz" : [],

        "cor_AxAy" : [], "cor_AxAz" : [], "cor_AyAz" : [], "cor_AxGx" : [], "cor_AxGy" : [], "cor_AxGz" : [],
        "cor_GxGy" : [], "cor_GxGz" : [], "cor_GyGz" : [], "cor_AyGx" : [], "cor_AyGy" : [], "cor_AyGz" : [],
        "cor_AzGx" : [], "cor_AzGy" : [], "cor_AzGz" : [],

        "skew_Ax" : [],"skew_Ay" : [],"skew_Az" : [],
        "skew_Gx"  : [],"skew_Gy"  : [],"skew_Gz"  : [],

        "kur_Ax" : [],"kur_Ay" : [],"kur_Az" : [],
        "kur_Gx"  : [],"kur_Gy"  : [],"kur_Gz"  : [],

        "sma_Ax" : [],"sma_Ay" : [],"sma_Az" : [],
        "sma_Gx"  : [],"sma_Gy"  : [],"sma_Gz"  : [],

        "mi_Acc" : [],"mi_G"  : [],

        "cf_Ax" : [], "cf_Ay" : [], "cf_Az" : [],
        "cf_Gx"  : [], "cf_Gy"  : [], "cf_Gz"  : [],

        "ir_Ax" : [], "ir_Ay" : [], "ir_Az" : [],
        "ir_Gx"  : [], "ir_Gy"  : [], "ir_Gz"  : [],

        "energy_Ax" : [],"energy_Ay" : [],"energy_Az" : [],
        "energy_Gx"  : [],"energy_Gy"  : [],"energy_Gz"  : [],

        "entropy_Ax" : [],"entropy_Ay" : [],"entropy_Az" : [],
        "entropy_Gx"  : [],"entropy_Gy"  : [],"entropy_Gz"  : [],
        }

    for i in range(count):
        file_name = f"{path2}/{str(i)}"
        with open(file_name + ".json", "r") as file:
            data = json.load(file)

    # Calculate features for each segmented sample.
        AccX = data["AccX"]
        AccY = data["AccY"]
        AccZ = data["AccZ"]
        GyX =  data["GyX"]
        GyY =  data["GyY"]
        GyZ =  data["GyZ"]

        # lpf data
        AccX_100 = gaussian_filter1d(AccX[50:150], sigma=0.5)
        AccY_100 = gaussian_filter1d(AccY[50:150], sigma=0.5)
        AccZ_100 = gaussian_filter1d(AccZ[50:150], sigma=0.5)
        GyX_100 = gaussian_filter1d(GyX[50:150], sigma=0.5)
        GyY_100 = gaussian_filter1d(GyY[50:150], sigma=0.5)
        GyZ_100 = gaussian_filter1d(GyZ[50:150], sigma=0.5)

        # max value array
        max_Ax.append(max(AccX_100))
        max_Ay.append(max(AccY_100))
        max_Az.append(max(AccZ_100))
        max_Gx.append(max(GyX_100))
        max_Gy.append(max(GyY_100))
        max_Gz.append(max(GyZ_100))

        # min value array
        min_Ax.append(min(AccX_100))
        min_Ay.append(min(AccY_100))
        min_Az.append(min(AccZ_100))
        min_Gx.append(min(GyX_100))
        min_Gy.append(min(GyY_100))
        min_Gz.append(min(GyZ_100))

        # mean value array
        mean_Ax.append(np.mean(AccX_100))
        mean_Ay.append(np.mean(AccY_100))
        mean_Az.append(np.mean(AccZ_100))
        mean_Gx.append(np.mean(GyX_100))
        mean_Gy.append(np.mean(GyY_100))
        mean_Gz.append(np.mean(GyZ_100))

        # RMS value array
        rms_Ax.append(calculate_rms(AccX_100))
        rms_Ay.append(calculate_rms(AccY_100))
        rms_Az.append(calculate_rms(AccZ_100))
        rms_Gx.append(calculate_rms(GyX_100))
        rms_Gy.append(calculate_rms(GyY_100))
        rms_Gz.append(calculate_rms(GyZ_100))

        # Mean absolute deviation (mad)
        mad_Ax.append(calculate_mad(AccX_100))
        mad_Ay.append(calculate_mad(AccY_100))
        mad_Az.append(calculate_mad(AccZ_100))
        mad_Gx.append(calculate_mad(GyX_100))
        mad_Gy.append(calculate_mad(GyY_100))
        mad_Gz.append(calculate_mad(GyZ_100))

        # median value array
        median_Ax.append(np.median(AccX_100))
        median_Ay.append(np.median(AccY_100))
        median_Az.append(np.median(AccZ_100))
        median_Gx.append(np.median(GyX_100))
        median_Gy.append(np.median(GyY_100))
        median_Gz.append(np.median(GyZ_100))

        # std value array
        std_Ax.append(np.std(AccX_100))
        std_Ay.append(np.std(AccY_100))
        std_Az.append(np.std(AccZ_100))
        std_Gx.append(np.std(GyX_100))
        std_Gy.append(np.std(GyY_100))
        std_Gz.append(np.std(GyZ_100))

        # variance value array
        var_Ax.append(np.var(AccX_100))
        var_Ay.append(np.var(AccY_100))
        var_Az.append(np.var(AccZ_100))
        var_Gx.append(np.var(GyX_100))
        var_Gy.append(np.var(GyY_100))
        var_Gz.append(np.var(GyZ_100))

        # covariance
        cov_AxAy.append(np.cov(AccX_100,AccY_100)[0, 1])
        cov_AxAz.append(np.cov(AccX_100,AccZ_100)[0, 1])
        cov_AyAz.append(np.cov(AccY_100,AccZ_100)[0, 1])
        cov_GxGy.append(np.cov(GyX_100,GyY_100)[0, 1])
        cov_GxGz.append(np.cov(GyX_100,GyZ_100)[0, 1])
        cov_GyGz.append(np.cov(GyY_100,GyZ_100)[0, 1])

        cov_AxGx.append(np.cov(AccX_100,GyX_100)[0, 1])
        cov_AxGy.append(np.cov(AccX_100,GyY_100)[0, 1])
        cov_AxGz.append(np.cov(AccX_100,GyZ_100)[0, 1])

        cov_AyGx.append(np.cov(AccY_100,GyX_100)[0, 1])
        cov_AyGy.append(np.cov(AccY_100,GyY_100)[0, 1])
        cov_AyGz.append(np.cov(AccY_100,GyZ_100)[0, 1])

        cov_AzGx.append(np.cov(AccZ_100,GyX_100)[0, 1])
        cov_AzGy.append(np.cov(AccZ_100,GyY_100)[0, 1])
        cov_AzGz.append(np.cov(AccZ_100,GyZ_100)[0, 1])

        # correlation coefficient
        cor_AxAy.append(np.corrcoef(AccX_100, AccY_100)[0, 1])
        cor_AxAz.append(np.corrcoef(AccX_100, AccZ_100)[0, 1])
        cor_AyAz.append(np.corrcoef(AccY_100, AccZ_100)[0, 1])
        cor_GxGy.append(np.corrcoef(GyX_100, GyY_100)[0, 1])
        cor_GxGz.append(np.corrcoef(GyX_100, GyZ_100)[0, 1])
        cor_GyGz.append(np.corrcoef(GyY_100, GyZ_100)[0, 1])

        cor_AxGx.append(np.corrcoef(AccX_100, GyX_100)[0, 1])
        cor_AxGy.append(np.corrcoef(AccX_100, GyY_100)[0, 1])
        cor_AxGz.append(np.corrcoef(AccX_100, GyZ_100)[0, 1])

        cor_AyGx.append(np.corrcoef(AccY_100, GyX_100)[0, 1])
        cor_AyGy.append(np.corrcoef(AccY_100, GyY_100)[0, 1])
        cor_AyGz.append(np.corrcoef(AccY_100, GyZ_100)[0, 1])

        cor_AzGx.append(np.corrcoef(AccZ_100, GyX_100)[0, 1])
        cor_AzGy.append(np.corrcoef(AccZ_100, GyY_100)[0, 1])
        cor_AzGz.append(np.corrcoef(AccZ_100, GyZ_100)[0, 1])

        # skewness value array
        skew_Ax.append(skew(AccX_100))
        skew_Ay.append(skew(AccY_100))
        skew_Az.append(skew(AccZ_100))
        skew_Gx.append(skew(GyX_100))
        skew_Gy.append(skew(GyY_100))
        skew_Gz.append(skew(GyZ_100))

        # kurtosis value array
        kur_Ax.append(kurtosis(AccX_100))
        kur_Ay.append(kurtosis(AccY_100))
        kur_Az.append(kurtosis(AccZ_100))
        kur_Gx.append(kurtosis(GyX_100))
        kur_Gy.append(kurtosis(GyY_100))
        kur_Gz.append(kurtosis(GyZ_100))

        # Signal magnitude area (sma)
        sma_Ax.append(calculate_sma(AccX_100))
        sma_Ay.append(calculate_sma(AccY_100))
        sma_Az.append(calculate_sma(AccZ_100))
        sma_Gx.append(calculate_sma(GyX_100))
        sma_Gy.append(calculate_sma(GyY_100))
        sma_Gz.append(calculate_sma(GyZ_100))

        # movement intensity (mi)
        mi_Acc.append(calculate_mi(AccX_100,AccY_100, AccZ_100))
        mi_G.append(calculate_mi(GyX_100,GyY_100, GyZ_100))

        # crest factor
        cf_Ax.append(max_Ax[i-1]/rms_Ax[i-1])
        cf_Ay.append(max_Ay[i-1]/rms_Ay[i-1])
        cf_Az.append(max_Az[i-1]/rms_Az[i-1])
        cf_Gx.append(max_Gx[i-1]/rms_Gx[i-1])
        cf_Gy.append(max_Gy[i-1]/rms_Gy[i-1])
        cf_Gz.append(max_Gz[i-1]/rms_Gz[i-1])

        # Interquartile Range
        ir_Ax.append(iqr(AccX_100))
        ir_Ay.append(iqr(AccY_100))
        ir_Az.append(iqr(AccZ_100))
        ir_Gx.append(iqr(GyX_100))
        ir_Gy.append(iqr(GyY_100))
        ir_Gz.append(iqr(GyZ_100))

        # Energy (fft)
        N = 50 # length of signal
        Fs = 100
        T = 1.0 / Fs

        AccX_f = np.fft.fft(AccX[75:125])
        AccY_f = np.fft.fft(AccY[75:125])
        AccZ_f = np.fft.fft(AccZ[75:125])
        GyX_f  = np.fft.fft(GyX[75:125])
        GyY_f  = np.fft.fft(GyY[75:125])
        GyZ_f  = np.fft.fft(GyZ[75:125])

        power_spectrum_AccX = np.abs(AccX_f)**2 / N; energy_AccX = np.sum(power_spectrum_AccX)
        power_spectrum_AccY = np.abs(AccY_f)**2 / N; energy_AccY = np.sum(power_spectrum_AccY)
        power_spectrum_AccZ = np.abs(AccZ_f)**2 / N; energy_AccZ = np.sum(power_spectrum_AccZ)
        power_spectrum_GyX = np.abs(GyX_f)**2 / N; energy_GyX = np.sum(power_spectrum_GyX)
        power_spectrum_GyY = np.abs(GyY_f)**2 / N; energy_GyY = np.sum(power_spectrum_GyY)
        power_spectrum_GyZ = np.abs(GyZ_f)**2 / N; energy_GyZ = np.sum(power_spectrum_GyZ)

        energy_Ax.append(energy_AccX)
        energy_Ay.append(energy_AccY)
        energy_Az.append(energy_AccZ)
        energy_Gx.append(energy_GyX)
        energy_Gy.append(energy_GyY)
        energy_Gz.append(energy_GyZ)

        ### Entropy
        # Normalize the PSD
        normalized_psd_AccX = power_spectrum_AccX/energy_AccX
        normalized_psd_AccY = power_spectrum_AccY/energy_AccY
        normalized_psd_AccZ = power_spectrum_AccZ/energy_AccZ
        normalized_psd_GyX = power_spectrum_GyX/energy_GyX
        normalized_psd_GyY = power_spectrum_GyX/energy_GyY
        normalized_psd_GyZ = power_spectrum_GyZ/energy_GyZ

        entropy_Ax.append(entropy(normalized_psd_AccX))
        entropy_Ay.append(entropy(normalized_psd_AccY))
        entropy_Az.append(entropy(normalized_psd_AccZ))
        entropy_Gx.append(entropy(normalized_psd_GyX))
        entropy_Gy.append(entropy(normalized_psd_GyY))
        entropy_Gz.append(entropy(normalized_psd_GyZ))

        frame.append(i)

        ### append features to json file
        features_data['max_Ax'] += [max_Ax[i]]
        features_data['max_Ay'] += [max_Ay[i]]
        features_data['max_Az'] += [max_Az[i]]
        features_data['max_Gx']  += [max_Gx[i]]
        features_data['max_Gy']  += [max_Gy[i]]
        features_data['max_Gz']  += [max_Gz[i]]

        features_data['min_Ax'] += [min_Ax[i]]
        features_data['min_Ay'] += [min_Ay[i]]
        features_data['min_Az'] += [min_Az[i]]
        features_data['min_Gx']  += [min_Gx[i]]
        features_data['min_Gy']  += [min_Gy[i]]
        features_data['min_Gz']  += [min_Gz[i]]

        features_data['mean_Ax'] += [mean_Ax[i]]
        features_data['mean_Ay'] += [mean_Ay[i]]
        features_data['mean_Az'] += [mean_Az[i]]
        features_data['mean_Gx']  += [mean_Gx[i]]
        features_data['mean_Gy']  += [mean_Gy[i]]
        features_data['mean_Gz']  += [mean_Gz[i]]

        features_data['rms_Ax'] += [rms_Ax[i]]
        features_data['rms_Ay'] += [rms_Ay[i]]
        features_data['rms_Az'] += [rms_Az[i]]
        features_data['rms_Gx'] += [rms_Gx[i]]
        features_data['rms_Gy'] += [rms_Gy[i]]
        features_data['rms_Gz'] += [rms_Gz[i]]

        features_data['mad_Ax'] += [mad_Ax[i]]
        features_data['mad_Ay'] += [mad_Ay[i]]
        features_data['mad_Az'] += [mad_Az[i]]
        features_data['mad_Gx']  += [mad_Gx[i]]
        features_data['mad_Gy']  += [mad_Gy[i]]
        features_data['mad_Gz']  += [mad_Gz[i]]

        features_data['median_Ax'] += [median_Ax[i]]
        features_data['median_Ay'] += [median_Ay[i]]
        features_data['median_Az'] += [median_Az[i]]
        features_data['median_Gx']  += [median_Gx[i]]
        features_data['median_Gy']  += [median_Gy[i]]
        features_data['median_Gz']  += [median_Gz[i]]

        features_data['std_Ax'] += [std_Ax[i]]
        features_data['std_Ay'] += [std_Ay[i]]
        features_data['std_Az'] += [std_Az[i]]
        features_data['std_Gx']  += [std_Gx[i]]
        features_data['std_Gy']  += [std_Gy[i]]
        features_data['std_Gz']  += [std_Gz[i]]

        features_data['var_Ax'] += [var_Ax[i]]
        features_data['var_Ay'] += [var_Ay[i]]
        features_data['var_Az'] += [var_Az[i]]
        features_data['var_Gx']  += [var_Gx[i]]
        features_data['var_Gy']  += [var_Gy[i]]
        features_data['var_Gz']  += [var_Gz[i]]

        features_data["cov_AxAy"] += [cov_AxAy[i]]
        features_data["cov_AxAz"] += [cov_AxAz[i]]
        features_data["cov_AyAz"] += [cov_AyAz[i]]
        features_data["cov_GxGy"] += [cov_GxGy[i]]
        features_data["cov_GxGz"] += [cov_GxGz[i]]
        features_data["cov_GyGz"] += [cov_GyGz[i]]
        features_data["cov_AxGx"] += [cov_AxGx[i]]
        features_data["cov_AxGy"] += [cov_AxGy[i]]
        features_data["cov_AxGz"] += [cov_AxGz[i]]
        features_data["cov_AyGx"] += [cov_AyGx[i]]
        features_data["cov_AyGy"] += [cov_AyGy[i]]
        features_data["cov_AyGz"] += [cov_AyGz[i]]
        features_data["cov_AzGx"] += [cov_AzGx[i]]
        features_data["cov_AzGy"] += [cov_AzGy[i]]
        features_data["cov_AzGz"] += [cov_AzGz[i]]

        features_data["cor_AxAy"] += [cor_AxAy[i]]
        features_data["cor_AxAz"] += [cor_AxAz[i]]
        features_data["cor_AyAz"] += [cor_AyAz[i]]
        features_data["cor_GxGy"] += [cor_GxGy[i]]
        features_data["cor_GxGz"] += [cor_GxGz[i]]
        features_data["cor_GyGz"] += [cor_GyGz[i]]
        features_data["cor_AxGx"] += [cor_AxGx[i]]
        features_data["cor_AxGy"] += [cor_AxGy[i]]
        features_data["cor_AxGz"] += [cor_AxGz[i]]
        features_data["cor_AyGx"] += [cor_AyGx[i]]
        features_data["cor_AyGy"] += [cor_AyGy[i]]
        features_data["cor_AyGz"] += [cor_AyGz[i]]
        features_data["cor_AzGx"] += [cor_AzGx[i]]
        features_data["cor_AzGy"] += [cor_AzGy[i]]
        features_data["cor_AzGz"] += [cor_AzGz[i]]

        features_data['skew_Ax'] += [skew_Ax[i]]
        features_data['skew_Ay'] += [skew_Ay[i]]
        features_data['skew_Az'] += [skew_Az[i]]
        features_data['skew_Gx']  += [skew_Gx[i]]
        features_data['skew_Gy']  += [skew_Gy[i]]
        features_data['skew_Gz']  += [skew_Gz[i]]

        features_data['kur_Ax'] += [kur_Ax[i]]
        features_data['kur_Ay'] += [kur_Ay[i]]
        features_data['kur_Az'] += [kur_Az[i]]
        features_data['kur_Gx']  += [kur_Gx[i]]
        features_data['kur_Gy']  += [kur_Gy[i]]
        features_data['kur_Gz']  += [kur_Gz[i]]

        features_data['sma_Ax'] += [sma_Ax[i]]
        features_data['sma_Ay'] += [sma_Ay[i]]
        features_data['sma_Az'] += [sma_Az[i]]
        features_data['sma_Gx'] += [sma_Gx[i]]
        features_data['sma_Gy'] += [sma_Gy[i]]
        features_data['sma_Gz'] += [sma_Gz[i]]

        features_data["mi_Acc"] += [mi_Acc[i]]
        features_data["mi_G"] += [mi_G[i]]

        features_data['cf_Ax'] += [cf_Ax[i]]
        features_data['cf_Ay'] += [cf_Ay[i]]
        features_data['cf_Az'] += [cf_Az[i]]
        features_data['cf_Gx'] += [cf_Gx[i]]
        features_data['cf_Gy'] += [cf_Gy[i]]
        features_data['cf_Gz'] += [cf_Gz[i]]

        features_data['ir_Ax'] += [ir_Ax[i]]
        features_data['ir_Ay'] += [ir_Ay[i]]
        features_data['ir_Az'] += [ir_Az[i]]
        features_data['ir_Gx'] += [ir_Gx[i]]
        features_data['ir_Gy'] += [ir_Gy[i]]
        features_data['ir_Gz'] += [ir_Gz[i]]

        features_data['energy_Ax'] += [energy_Ax[i]]
        features_data['energy_Ay'] += [energy_Ay[i]]
        features_data['energy_Az'] += [energy_Az[i]]
        features_data['energy_Gx']  += [energy_Gx[i]]
        features_data['energy_Gy']  += [energy_Gy[i]]
        features_data['energy_Gz']  += [energy_Gz[i]]

        features_data['entropy_Ax'] += [entropy_Ax[i]]
        features_data['entropy_Ay'] += [entropy_Ay[i]]
        features_data['entropy_Az'] += [entropy_Az[i]]
        features_data['entropy_Gx']  += [entropy_Gx[i]]
        features_data['entropy_Gy']  += [entropy_Gy[i]]
        features_data['entropy_Gz']  += [entropy_Gz[i]]

        with open( f"{path}features.json", 'w') as f:
            json.dump(features_data, f, indent=4)

    ########################### append feature to array to predict ############################
    X = [];y = []

    Max_Ax = [];Max_Ay = [];Max_Az = [];Max_Gx = [];Max_Gy = [];Max_Gz = []
    Min_Ax = [];Min_Ay = [];Min_Az = [];Min_Gx = [];Min_Gy = [];Min_Gz = []
    Mean_Ax = [];Mean_Ay = [];Mean_Az = [];Mean_Gx = [];Mean_Gy = [];Mean_Gz = []
    Rms_Ax = [];Rms_Ay = [];Rms_Az = [];Rms_Gx = [];Rms_Gy = [];Rms_Gz = []
    Mad_Ax = [];Mad_Ay = [];Mad_Az = [];Mad_Gx = [];Mad_Gy = [];Mad_Gz = []
    Median_Ax = [];Median_Ay = [];Median_Az = [];Median_Gx = [];Median_Gy = [];Median_Gz = []
    Std_Ax = [];Std_Ay = [];Std_Az = [];Std_Gx = [];Std_Gy = [];Std_Gz = []
    Var_Ax = [];Var_Ay = [];Var_Az = [];Var_Gx = [];Var_Gy = [];Var_Gz = []
    Cov_AxAy = []; Cov_AxAz = []; Cov_AyAz = []; Cov_AxGx = []; Cov_AxGy = []; Cov_AxGz = []
    Cov_GxGy = []; Cov_GxGz = []; Cov_GyGz = []; Cov_AyGx = []; Cov_AyGy = []; Cov_AyGz = []
    Cov_AzGx = []; Cov_AzGy = []; Cov_AzGz = []
    Cor_AxAy = []; Cor_AxAz = []; Cor_AyAz = []; Cor_AxGx = []; Cor_AxGy = []; Cor_AxGz = []
    Cor_GxGy = []; Cor_GxGz = []; Cor_GyGz = []; Cor_AyGx = []; Cor_AyGy = []; Cor_AyGz = []
    Cor_AzGx = []; Cor_AzGy = []; Cor_AzGz = []
    Skew_Ax = [];Skew_Ay = [];Skew_Az = [];Skew_Gx = [];Skew_Gy = [];Skew_Gz = []
    Kur_Ax = [];Kur_Ay = [];Kur_Az = [];Kur_Gx = [];Kur_Gy = [];Kur_Gz = []
    Sma_Ax = [];Sma_Ay = [];Sma_Az = [];Sma_Gx = [];Sma_Gy = [];Sma_Gz = []
    Mi_Acc = [];Mi_G = []
    Cf_Ax = [];Cf_Ay = [];Cf_Az = [];Cf_Gx = [];Cf_Gy = [];Cf_Gz = []
    Ir_Ax = [];Ir_Ay = [];Ir_Az = [];Ir_Gx = [];Ir_Gy = [];Ir_Gz = []
    Energy_Ax = [];Energy_Ay = [];Energy_Az = [];Energy_Gx = [];Energy_Gy = [];Energy_Gz = []
    Entropy_Ax = []; Entropy_Ay = []; Entropy_Az = []; Entropy_Gx = []; Entropy_Gy = []; Entropy_Gz = []

    ### import file data
    with open( f"{path}features.json", 'r') as file:
        data = json.load(file)

    def append_data_feature(number_of_data):
        for m in range(number_of_data):

            Max_Ax.append(max_Ax[m])
            Max_Ay.append(max_Ay[m])
            Max_Az.append(max_Az[m])
            Max_Gx.append(max_Gx[m])
            Max_Gy.append(max_Gy[m])
            Max_Gz.append(max_Gz[m])

            Min_Ax.append(min_Ax[m])
            Min_Ay.append(min_Ay[m])
            Min_Az.append(min_Az[m])
            Min_Gx.append(min_Gx[m])
            Min_Gy.append(min_Gy[m])
            Min_Gz.append(min_Gz[m])

            Mean_Ax.append(mean_Ax[m])
            Mean_Ay.append(mean_Ay[m])
            Mean_Az.append(mean_Az[m])
            Mean_Gx.append(mean_Gx[m])
            Mean_Gy.append(mean_Gy[m])
            Mean_Gz.append(mean_Gz[m])

            Rms_Ax.append(rms_Ax[m])
            Rms_Ay.append(rms_Ay[m])
            Rms_Az.append(rms_Az[m])
            Rms_Gx.append(rms_Gx[m])
            Rms_Gy.append(rms_Gy[m])
            Rms_Gz.append(rms_Gz[m])

            Mad_Ax.append(mad_Ax[m])
            Mad_Ay.append(mad_Ay[m])
            Mad_Az.append(mad_Az[m])
            Mad_Gx.append(mad_Gx[m])
            Mad_Gy.append(mad_Gy[m])
            Mad_Gz.append(mad_Gz[m])

            Median_Ax.append(median_Ax[m])
            Median_Ay.append(median_Ay[m])
            Median_Az.append(median_Az[m])
            Median_Gx.append(median_Gx[m])
            Median_Gy.append(median_Gy[m])
            Median_Gz.append(median_Gz[m])

            Std_Ax.append(std_Ax[m])
            Std_Ay.append(std_Ay[m])
            Std_Az.append(std_Az[m])
            Std_Gx.append(std_Gx[m])
            Std_Gy.append(std_Gy[m])
            Std_Gz.append(std_Gz[m])

            Var_Ax.append(var_Ax[m])
            Var_Ay.append(var_Ay[m])
            Var_Az.append(var_Az[m])
            Var_Gx.append(var_Gx[m])
            Var_Gy.append(var_Gy[m])
            Var_Gz.append(var_Gz[m])

            Cov_AxAy.append(cov_AxAy[m])
            Cov_AxAz.append(cov_AxAz[m])
            Cov_AxGx.append(cov_AxGx[m])
            Cov_AxGy.append(cov_AxGy[m])
            Cov_AxGz.append(cov_AxGz[m])
            Cov_AyAz.append(cov_AyAz[m])
            Cov_AyGx.append(cov_AyGx[m])
            Cov_AyGy.append(cov_AyGy[m])
            Cov_AyGz.append(cov_AyGz[m])
            Cov_AzGx.append(cov_AzGx[m])
            Cov_AzGy.append(cov_AzGy[m])
            Cov_AzGz.append(cov_AzGz[m])
            Cov_GxGy.append(cov_GxGy[m])
            Cov_GxGz.append(cov_GxGz[m])
            Cov_GyGz.append(cov_GyGz[m])

            Cor_AxAy.append(cor_AxAy[m])
            Cor_AxAz.append(cor_AxAz[m])
            Cor_AxGx.append(cor_AxGx[m])
            Cor_AxGy.append(cor_AxGy[m])
            Cor_AxGz.append(cor_AxGz[m])
            Cor_AyAz.append(cor_AyAz[m])
            Cor_AyGx.append(cor_AyGx[m])
            Cor_AyGy.append(cor_AyGy[m])
            Cor_AyGz.append(cor_AyGz[m])
            Cor_AzGx.append(cor_AzGx[m])
            Cor_AzGy.append(cor_AzGy[m])
            Cor_AzGz.append(cor_AzGz[m])
            Cor_GxGy.append(cor_GxGy[m])
            Cor_GxGz.append(cor_GxGz[m])
            Cor_GyGz.append(cor_GyGz[m])

            Skew_Ax.append(skew_Ax[m])
            Skew_Ay.append(skew_Ay[m])
            Skew_Az.append(skew_Az[m])
            Skew_Gx.append(skew_Gx[m])
            Skew_Gy.append(skew_Gy[m])
            Skew_Gz.append(skew_Gz[m])

            Kur_Ax.append(kur_Ax[m])
            Kur_Ay.append(kur_Ay[m])
            Kur_Az.append(kur_Az[m])
            Kur_Gx.append(kur_Gx[m])
            Kur_Gy.append(kur_Gy[m])
            Kur_Gz.append(kur_Gz[m])

            Sma_Ax.append(sma_Ax[m])
            Sma_Ay.append(sma_Ay[m])
            Sma_Az.append(sma_Az[m])
            Sma_Gx.append(sma_Gx[m])
            Sma_Gy.append(sma_Gy[m])
            Sma_Gz.append(sma_Gz[m])

            Mi_Acc.append(mi_Acc[m])
            Mi_G.append(mi_G[m])

            Cf_Ax.append(cf_Ax[m])
            Cf_Ay.append(cf_Ay[m])
            Cf_Az.append(cf_Az[m])
            Cf_Gx.append(cf_Gx[m])
            Cf_Gy.append(cf_Gy[m])
            Cf_Gz.append(cf_Gz[m])

            Ir_Ax.append(ir_Ax[m])
            Ir_Ay.append(ir_Ay[m])
            Ir_Az.append(ir_Az[m])
            Ir_Gx.append(ir_Gx[m])
            Ir_Gy.append(ir_Gy[m])
            Ir_Gz.append(ir_Gz[m])

            Energy_Ax.append(energy_Ax[m])
            Energy_Ay.append(energy_Ay[m])
            Energy_Az.append(energy_Az[m])
            Energy_Gx.append(energy_Gx[m])
            Energy_Gy.append(energy_Gy[m])
            Energy_Gz.append(energy_Gz[m])

            Entropy_Ax.append(entropy_Ax[m])
            Entropy_Ay.append(entropy_Ay[m])
            Entropy_Az.append(entropy_Az[m])
            Entropy_Gx.append(entropy_Gx[m])
            Entropy_Gy.append(entropy_Gy[m])
            Entropy_Gz.append(entropy_Gz[m])

    ### Access the data from the file
    max_Ax = data["max_Ax"]
    max_Ay = data["max_Ay"]
    max_Az = data["max_Az"]
    max_Gx = data["max_Gx"]
    max_Gy = data["max_Gy"]
    max_Gz = data["max_Gz"]

    min_Ax = data["min_Ax"]
    min_Ay = data["min_Ay"]
    min_Az = data["min_Az"]
    min_Gx = data["min_Gx"]
    min_Gy = data["min_Gy"]
    min_Gz = data["min_Gz"]

    mean_Ax = data["mean_Ax"]
    mean_Ay = data["mean_Ay"]
    mean_Az = data["mean_Az"]
    mean_Gx = data["mean_Gx"]
    mean_Gy = data["mean_Gy"]
    mean_Gz = data["mean_Gz"]

    rms_Ax = data["rms_Ax"]
    rms_Ay = data["rms_Ay"]
    rms_Az = data["rms_Az"]
    rms_Gx = data["rms_Gx"]
    rms_Gy = data["rms_Gy"]
    rms_Gz = data["rms_Gz"]

    mad_Ax = data["mad_Ax"]
    mad_Ay = data["mad_Ay"]
    mad_Az = data["mad_Az"]
    mad_Gx = data["mad_Gx"]
    mad_Gy = data["mad_Gy"]
    mad_Gz = data["mad_Gz"]

    median_Ax = data["median_Ax"]
    median_Ay = data["median_Ay"]
    median_Az = data["median_Az"]
    median_Gx = data["median_Gx"]
    median_Gy = data["median_Gy"]
    median_Gz = data["median_Gz"]

    std_Ax = data["std_Ax"]
    std_Ay = data["std_Ay"]
    std_Az = data["std_Az"]
    std_Gx = data["std_Gx"]
    std_Gy = data["std_Gy"]
    std_Gz = data["std_Gz"]

    var_Ax = data["mean_Ax"]
    var_Ay = data["mean_Ay"]
    var_Az = data["mean_Az"]
    var_Gx = data["mean_Gx"]
    var_Gy = data["mean_Gy"]
    var_Gz = data["mean_Gz"]

    cov_AxAy = data["cov_AxAy"]
    cov_AxAz = data["cov_AxAz"]
    cov_AxGx = data["cov_AxGx"]
    cov_AxGy = data["cov_AxGy"]
    cov_AxGz = data["cov_AxGz"]
    cov_AyAz = data["cov_AyAz"]
    cov_AyGx = data["cov_AyGx"]
    cov_AyGy = data["cov_AyGy"]
    cov_AyGz = data["cov_AyGz"]
    cov_AzGx = data["cov_AzGx"]
    cov_AzGy = data["cov_AzGy"]
    cov_AzGz = data["cov_AzGz"]
    cov_GxGy = data["cov_GxGy"]
    cov_GxGz = data["cov_GxGz"]
    cov_GyGz = data["cov_GyGz"]

    cor_AxAy = data["cor_AxAy"]
    cor_AxAz = data["cor_AxAz"]
    cor_AxGx = data["cor_AxGx"]
    cor_AxGy = data["cor_AxGy"]
    cor_AxGz = data["cor_AxGz"]
    cor_AyAz = data["cor_AyAz"]
    cor_AyGx = data["cor_AyGx"]
    cor_AyGy = data["cor_AyGy"]
    cor_AyGz = data["cor_AyGz"]
    cor_AzGx = data["cor_AzGx"]
    cor_AzGy = data["cor_AzGy"]
    cor_AzGz = data["cor_AzGz"]
    cor_GxGy = data["cor_GxGy"]
    cor_GxGz = data["cor_GxGz"]
    cor_GyGz = data["cor_GyGz"]

    skew_Ax = data["skew_Ax"]
    skew_Ay = data["skew_Ay"]
    skew_Az = data["skew_Az"]
    skew_Gx = data["skew_Gx"]
    skew_Gy = data["skew_Gy"]
    skew_Gz = data["skew_Gz"]

    kur_Ax = data["kur_Ax"]
    kur_Ay = data["kur_Ay"]
    kur_Az = data["kur_Az"]
    kur_Gx = data["kur_Gx"]
    kur_Gy = data["kur_Gy"]
    kur_Gz = data["kur_Gz"]

    sma_Ax = data["sma_Ax"]
    sma_Ay = data["sma_Ay"]
    sma_Az = data["sma_Az"]
    sma_Gx = data["sma_Gx"]
    sma_Gy = data["sma_Gy"]
    sma_Gz = data["sma_Gz"]

    mi_Acc = data["mi_Acc"]
    mi_G = data["mi_G"]

    cf_Ax = data["cf_Ax"]
    cf_Ay = data["cf_Ay"]
    cf_Az = data["cf_Az"]
    cf_Gx = data["cf_Gx"]
    cf_Gy = data["cf_Gy"]
    cf_Gz = data["cf_Gz"]

    ir_Ax = data["ir_Ax"]
    ir_Ay = data["ir_Ay"]
    ir_Az = data["ir_Az"]
    ir_Gx = data["ir_Gx"]
    ir_Gy = data["ir_Gy"]
    ir_Gz = data["ir_Gz"]

    energy_Ax = data["energy_Ax"]
    energy_Ay = data["energy_Ay"]
    energy_Az = data["energy_Az"]
    energy_Gx = data["energy_Gx"]
    energy_Gy = data["energy_Gy"]
    energy_Gz = data["energy_Gz"]

    entropy_Ax = data["entropy_Ax"]
    entropy_Ay = data["entropy_Ay"]
    entropy_Az = data["entropy_Az"]
    entropy_Gx = data["entropy_Gx"]
    entropy_Gy = data["entropy_Gy"]
    entropy_Gz = data["entropy_Gz"]

    append_data_feature(len(max_Gy))

    ### append feature from sample to an array
    for i in range(len(Max_Gy)):
        X.append([

                    Max_Ax[i],
                    Max_Ay[i],
                    Max_Az[i],
                    Max_Gx[i],
                    Max_Gy[i],
                    Max_Gz[i],

                    Min_Ax[i],
                    Min_Ay[i],
                    Min_Az[i],
                    Min_Gx[i],
                    Min_Gy[i],
                    Min_Gz[i],

                    Mean_Ax[i],
                    Mean_Ay[i],
                    Mean_Az[i],
                    Mean_Gx[i],
                    Mean_Gy[i],
                    Mean_Gz[i],

                    Rms_Ax[i],
                    Rms_Ay[i],
                    Rms_Az[i],
                    Rms_Gx[i],
                    Rms_Gy[i],
                    Rms_Gz[i],

                    Mad_Ax[i],
                    Mad_Ay[i],
                    Mad_Az[i],
                    Mad_Gx[i],
                    Mad_Gy[i],
                    Mad_Gz[i],

                    Median_Ax[i],
                    Median_Ay[i],
                    Median_Az[i],
                    Median_Gx[i],
                    Median_Gy[i],
                    Median_Gz[i],

                    Std_Ax[i],
                    Std_Ay[i],
                    Std_Az[i],
                    Std_Gx[i],
                    Std_Gy[i],
                    Std_Gz[i],

                    Var_Ax[i],
                    Var_Ay[i],
                    Var_Az[i],
                    Var_Gx[i],
                    Var_Gy[i],
                    Var_Gz[i],

                    Cov_AxAy[i],
                    Cov_AxAz[i],
                    Cov_AxGx[i],
                    Cov_AxGy[i],
                    Cov_AxGz[i],
                    Cov_AyAz[i],
                    Cov_AyGx[i],
                    Cov_AyGy[i],
                    Cov_AyGz[i],
                    Cov_AzGx[i],
                    Cov_AzGy[i],
                    Cov_AzGz[i],
                    Cov_GxGy[i],
                    Cov_GxGz[i],
                    Cov_GyGz[i],

                    Cor_AxAy[i],
                    Cor_AxAz[i],
                    Cor_AxGx[i],
                    Cor_AxGy[i],
                    Cor_AxGz[i],
                    Cor_AyAz[i],
                    Cor_AyGx[i],
                    Cor_AyGy[i],
                    Cor_AyGz[i],
                    Cor_AzGx[i],
                    Cor_AzGy[i],
                    Cor_AzGz[i],
                    Cor_GxGy[i],
                    Cor_GxGz[i],
                    Cor_GyGz[i],

                    Skew_Ax[i],
                    Skew_Ay[i],
                    Skew_Az[i],
                    Skew_Gx[i],
                    Skew_Gy[i],
                    Skew_Gz[i],

                    Kur_Ax[i],
                    Kur_Ay[i],
                    Kur_Az[i],
                    Kur_Gx[i],
                    Kur_Gy[i],
                    Kur_Gz[i],

                    Sma_Ax[i],
                    Sma_Ay[i],
                    Sma_Az[i],
                    Sma_Gx[i],
                    Sma_Gy[i],
                    Sma_Gz[i],

                    Mi_Acc[i],
                    Mi_G[i],

                    Cf_Ax[i],
                    Cf_Ay[i],
                    Cf_Az[i],
                    Cf_Gx[i],
                    Cf_Gy[i],
                    Cf_Gz[i],

                    Ir_Ax[i],
                    Ir_Ay[i],
                    Ir_Az[i],
                    Ir_Gx[i],
                    Ir_Gy[i],
                    Ir_Gz[i],

                    Energy_Ax[i],
                    Energy_Ay[i],
                    Energy_Az[i],
                    Energy_Gx[i],
                    Energy_Gy[i],
                    Energy_Gz[i],

                    Entropy_Ax[i],
                    Entropy_Ay[i],
                    Entropy_Az[i],
                    Entropy_Gx[i],
                    Entropy_Gy[i],
                    Entropy_Gz[i],
                    ])

    # load model from pickle file
    # svm_model.pkl applies standardization and PCA itself, so it takes the
    # 122 raw features directly
    with open(SVM_MODEL_PATH, 'rb') as file:
        model = pickle.load(file)

    X = np.array(X)

    # stop with a clear message if the model expects a different input size
    # (e.g. a bare SVM saved without its scaler and PCA)
    n_expected = getattr(model, 'n_features_in_', X.shape[1])
    if n_expected != X.shape[1]:
        raise ValueError(
            f"svm_model.pkl expects {n_expected} input features, but {X.shape[1]} were "
            "extracted. Save the scaler, PCA and SVM together as one scikit-learn Pipeline."
        )

    # print shot count
    print("Shot count: ", count)

    # evaluate model
    y_predict = model.predict(X)
    print(y_predict)

if __name__ == "__main__":
    main()