import json
import numpy as np
import sys
from pathlib import Path
from scipy.stats import skew, kurtosis
from scipy.stats import iqr, entropy
from itertools import combinations

# Path to the dataset JSON file
structure_dir = "/home/pnguyen/Workspace/IU-VNU/THESIS/Thesis_Devevelopment/construct_data/dataset_structure.json"
dataset_dir = "/home/pnguyen/Workspace/IU-VNU/THESIS/Thesis_Devevelopment/construct_data/badminton_stroke_dataset.json"

sampling_rate = 100  # Hz
N = 50 # Number of sample for FFT analysis

def count_files(directory):
    try:
        path = Path(directory)
        # Count only files (not directories)
        file_count = sum(1 for item in path.iterdir() if item.is_file() and item.suffix == '.json')
        return file_count
    except Exception:
        print(f"Directory '{directory}' not found.")
        return 0


def stroke_type_encoder(directory):
    stroke_type_map = {
        "Clear_fake": 1,
        "Slice_fake": 2,
        "Drive_fake": 3,
        "Smash_fake": 4,
        "Clear_real": 5,
        "Slice_real": 6,
        "Drive_real": 7,
        "Smash_real": 8
    }

    for pattern, value in stroke_type_map.items():
        if pattern in directory:
            return pattern, value
    return  # Default value if no pattern matches


def get_imu_data(file_name):
    try:
        with open(file_name, 'r') as file:
            data = json.load(file)
    except FileNotFoundError:
        print(f"File '{file_name}' not found.")
        return None
    else:
        # Access the IMU data
        AccX = data["AccX"]
        AccY = data["AccY"]
        AccZ = data["AccZ"]
        GyX =  data["GyX"]
        GyY =  data["GyY"]
        GyZ =  data["GyZ"]
        frame= data["frame"]

        return AccX, AccY, AccZ, GyX, GyY, GyZ


def initialized_features():
    features = {}

    # Statistical features for each axis
    axes = ['Ax', 'Ay', 'Az', 'Gx', 'Gy', 'Gz']
    stats = ['max', 'min', 'mean', 'rms', 'mad', 'median', 'std', 'var', 
             'skew', 'kur', 'sma', 'cf', 'ir', 'energy', 'entropy']
    
    for stat in stats:
        for axis in axes:
            features [f"{stat}_{axis}"] = 0.0

    
    # Covariance features between axes
    covariance_pair = [
        'AxAy', 'AxAz', 'AyAz', 'AxGx', 'AxGy', 'AxGz',
        'GxGy', 'GxGz', 'GyGz', 'AyGx', 'AyGy', 'AyGz',
        'AzGx', 'AzGy', 'AzGz'
    ]

    for pair in covariance_pair:
        features[f"cov_{pair}"] = 0.0
        features[f"cor_{pair}"] = 0.0

    # Special features
    features["mi_Acc"] = 0.0
    features["mi_G"] = 0.0

    return features


def energy(axis_data):
    """
    Calculate energy of the signal using FFT. 
    
    Args:
        axis_data: list or numpy array of signal data for one axis
    
    Returns:
        Energy of the signal
    """
    # Compute FFT
    fft = np.fft.fft(axis_data[25:75]) # Window size of 50
    # Compute normalized power spectral density (PSD)
    psd = np.abs(fft)**2 / N
    # Compute energy
    energy = np.sum(psd)
    return energy

def entropy(axis_data):
    """
    Calculate entropy of the signal using FFT. 
    
    Args:
        axis_data: list or numpy array of signal data for one axis
    
    Returns:
        Entropy of the signal
    """
    # Compute FFT
    fft = np.fft.fft(axis_data[25:75]) # Window size of 50
    # Compute normalized power spectral density (PSD)
    psd = np.abs(fft)**2 / N
    energy = np.sum(psd)
    normalized_psd = psd / energy
    # Compute entropy
    ent = -np.sum(normalized_psd * np.log(normalized_psd + 1e-10))  # Adding a small value to avoid log(0)
    return ent


def mi(x, y, z):
    """ Calculate the magnitude of the 3D vector and return its sum. """
    magnitude = np.sqrt(x**2 + y**2 + z**2)
    return np.sum(magnitude)


def calculate_features(Ax, Ay, Az, Gx, Gy, Gz):
    """
    Calculate features from IMU data.
    
    Args:
        AccX, AccY, AccZ: Accelerometer data for X, Y, Z axes
        GyX, GyY, GyZ: Gyroscope data for X, Y, Z axes
    
    Returns:
        A dictionary of calculated features
    """
    # Standardize data
    Ax = np.array(Ax[50:150])
    Ay = np.array(Ay[50:150])
    Az = np.array(Az[50:150])
    Gx = np.array(Gx[50:150])
    Gy = np.array(Gy[50:150])
    Gz = np.array(Gz[50:150])

    # create feature dictionary
    features = {}
    
    # Statistical features for each axis
    axes = ['Ax', 'Ay', 'Az', 'Gx', 'Gy', 'Gz']
    stats = ['max', 'min', 'mean', 'rms', 'mad', 'median', 'std', 'var', 
             'skew', 'kur', 'sma', 'cf', 'ir']
    fregs = ['energy', 'entropy']
    integrated = ['mi_Acc', 'mi_G']

    for stat in stats:
        for axis in axes:
            if stat == 'max':
                features[f"{stat}_{axis}"] = np.max(eval(axis))
            elif stat == 'min':
                features[f"{stat}_{axis}"] = np.min(eval(axis))
            elif stat == 'mean':
                features[f"{stat}_{axis}"] = np.mean(eval(axis))
            elif stat == 'rms':
                features[f"{stat}_{axis}"] = np.sqrt(np.mean(np.square(eval(axis))))
            elif stat == 'mad':
                features[f"{stat}_{axis}"] = np.mean(np.abs(eval(axis) - np.mean(eval(axis))))
            elif stat == 'median':
                features[f"{stat}_{axis}"] = np.median(eval(axis))
            elif stat == 'std':
                features[f"{stat}_{axis}"] = np.std(eval(axis))
            elif stat == 'var':
                features[f"{stat}_{axis}"] = np.var(eval(axis))
            elif stat == 'skew':
                features[f"{stat}_{axis}"] = skew(eval(axis))
            elif stat == 'kur':
                features[f"{stat}_{axis}"] = kurtosis(eval(axis))
            elif stat == 'sma':
                features[f"{stat}_{axis}"] = np.sum(np.abs(eval(axis))) / len(eval(axis))
            elif stat == 'cf':
                features[f"{stat}_{axis}"] = np.max(np.abs(eval(axis))) / np.sqrt(np.mean(np.square(eval(axis))))
            elif stat == 'ir':
                features[f"{stat}_{axis}"] = iqr(eval(axis))
        
    for freg in fregs: 
        for axis in axes:
            if freg == 'energy':
                features[f"{freg}_{axis}"] = energy(eval(axis))
            elif freg == 'entropy':
                features[f"{freg}_{axis}"] = entropy(eval(axis))
    
    for integ in integrated:
            if integ == 'mi_Acc':
                features[f"{integ}"] = mi(Ax, Ay, Az)
            elif integ == 'mi_G':
                features[f"{integ}"] = mi(Gx, Gy, Gz)
    
    axes_pair = list(combinations(axes, 2))
    for pair in axes_pair:
        features[f"cov_{pair[0]}{pair[1]}"] = np.cov(eval(pair[0]), eval(pair[1]))[0][1]
        features[f"cor_{pair[0]}{pair[1]}"] = np.corrcoef(eval(pair[0]), eval(pair[1]))[0][1]
    
    return features


def dictionary_parsing(file_name, dataset_dict, sample_id, stroke_type, label):
    """
    Parse IMU data and add it to a dictionary structure.
    
    Args:
        file_name: path to the IMU data file
        dataset: existing dataset structure
        sample_id: unique identifier for this sample
        stroke_type: type of badminton stroke
        label: numeric label for the stroke

    Returns:
        Updated dictionary
    """
    # Unpack IMU data
    imu_data = get_imu_data(file_name)
    if imu_data is None:
        print(f"IMU data for sample {sample_id} is missing.")
        return
    Ax, Ay, Az, Gx, Gy, Gz = imu_data

    # Create new sample
    new_sample = {
        "sample_id": sample_id,
        "stroke_type": stroke_type,
        "label": label,
        "IMU_data_sequences": {
            "Ax": Ax,
            "Ay": Ay,
            "Az": Az,
            "Gx": Gx,
            "Gy": Gy,
            "Gz": Gz
        },
        "extracted_features": calculate_features(Ax, Ay, Az, Gx, Gy, Gz)
    }

    # Add new sample to the dataset
    dataset_dict["samples"].append(new_sample)

    return dataset_dict


def dataset_writing(directory, file_count, stroke_type, label):
    """
    Write IMU data from files in a directory to a dataset JSON file.

    Args:
        directory: path to the directory containing IMU data files
        file_count: number of files in the directory
        stroke_type: type of badminton stroke
        label: numeric label for the stroke
    """
    # Load existing dataset
    try:
        with open(dataset_dir, 'r') as f:
            dataset_dict = json.load(f)
    except FileNotFoundError:
        print(f"Dataset directory {dataset_dir} not found.")
        return
    
    # Current amount of samples
    current_samples = len(dataset_dict["samples"]) - 1  # Exclude the initial empty sample
    print(f"Current samples in dataset: {current_samples}")

    # Loop through files in the directory
    for sample in range(1, file_count + 1):
        file_name = f"{directory}/{sample}_{label}.json"

        # Append new sample to dictionary
        updated_dict = dictionary_parsing(file_name, dataset_dict, current_samples + sample, stroke_type, label)
        print(f"Processed sample {sample+current_samples} for stroke type '{stroke_type}' with label {label}")
    
    if updated_dict is None:
        print("No new samples were added to the dataset.")
        return
    else:
        # Update total sample count
        dataset_dict["dataset_info"]["total_samples"] = len(dataset_dict["samples"]) - 1

        # Update stroke distribution
        for stroke_type in dataset_dict["dataset_info"]["stroke_types"]:
            dataset_dict["dataset_info"]["stroke_distribution"][stroke_type] = \
            sum(1 for sample in dataset_dict["samples"] if sample["stroke_type"] == stroke_type)

        # Write updated dictionary back to the dataset file
        with open(dataset_dir, 'w') as f:
            json.dump(updated_dict, f, indent=2)


def main():
    argc = len(sys.argv) # Number of arguments
    argv = sys.argv # Arguments list
    
    if argc == 2:
        directory = argv[1]
    else:
        print("Usage: python merge_data.py <directory>")
        return
    
    # Count files in the specified directory
    file_count = count_files(directory)
    
    # Find stroke type and label
    stroke_type, label = stroke_type_encoder(directory)

    # Write information to dataset
    dataset_writing(directory, file_count, stroke_type, label)

    
if __name__ == "__main__":
    main()