import json
import numpy as np
import pickle
from datetime import datetime
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# File paths
DATASET_FILE = "dataset/badminton_stroke_dataset.json"
MODEL_FILE = "ClassicalML/models/svm_model.pkl"
MODEL_INFO_FILE = "ClassicalML/models/svm_info.txt"

# Load dataset
print("Loading dataset...")
with open(DATASET_FILE, 'r') as file:
    data = json.load(file)

# Extract features and labels
X = []
y = []

for sample in data['samples'][1:]:
    features = list(sample['extracted_features'].values())
    X.append(features)
    y.append(sample['label'])

# Convert to numpy arrays
X = np.array(X)
y = np.array(y)

print(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")

# Split dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=None)

print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")

# Create and train the classifier
print("Training model...")
clf = make_pipeline(
    StandardScaler(), 
    PCA(n_components=30), 
    SVC(kernel='rbf', C=1, gamma='scale')
)

clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# Display results
print(f"Accuracy: {accuracy:.4f}")
print(f"\nConfusion Matrix:\n{conf_matrix}")
print(f"\nClassification Report:\n{class_report}")

# Save model
print(f"\nSaving model to {MODEL_FILE}...")
with open(MODEL_FILE, 'wb') as file:
    pickle.dump(clf, file)

# Save model info
print(f"Saving model info to {MODEL_INFO_FILE}...")
model_info = {
    "model_name": "SVM with RBF Kernel",
    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "hyperparameters": {
        "kernel": "rbf",
        "C": 1,
        "gamma": "scale",
        "pca_components": 30,
        "scaler": "StandardScaler"
    },
    "dataset_info": {
        "total_samples": X.shape[0],
        "num_features": X.shape[1],
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "test_size": 0.25
    },
    "performance": {
        "accuracy": float(accuracy),
        "confusion_matrix": conf_matrix.tolist(),
    },
    "classification_report": class_report
}

with open(MODEL_INFO_FILE, 'w') as file:
    file.write(f"{'='*60}\n")
    file.write(f"Model Information\n")
    file.write(f"{'='*60}\n\n")
    file.write(f"Model Name: {model_info['model_name']}\n")
    file.write(f"Timestamp: {model_info['timestamp']}\n\n")
    
    file.write(f"Hyperparameters:\n")
    for key, value in model_info['hyperparameters'].items():
        file.write(f"  - {key}: {value}\n")
    file.write("\n")
    
    file.write(f"Dataset Information:\n")
    for key, value in model_info['dataset_info'].items():
        file.write(f"  - {key}: {value}\n")
    file.write("\n")
    
    file.write(f"Performance Metrics:\n")
    file.write(f"  - Accuracy: {accuracy:.4f}\n\n")
    
    file.write(f"Confusion Matrix:\n{conf_matrix}\n\n")
    file.write(f"Classification Report:\n{class_report}\n")

print("Done!")


# accuracy_arr = []
# g_arr = np.linspace(0.00001,0.5,1000)

# for i in range(1000):
#     g = g_arr[i]
#     clf = make_pipeline(StandardScaler(), PCA(n_components=22),SVC(kernel='rbf',C=19, gamma=g))

#     # Train the classifier
#     clf.fit(X_train, y_train)

#     # Make predictions on the test set
#     y_pred = clf.predict(X_test)

#     # Evaluate the accuracy of the model
#     accuracy = accuracy_score(y_test, y_pred)
#     accuracy_arr.append(accuracy)
#     # g_arr.append(g)

# plt.plot(g_arr,accuracy_arr)
# plt.xlabel('gamma')
# plt.ylabel('Accuracy')
# plt.title('SVM')
# plt.show()