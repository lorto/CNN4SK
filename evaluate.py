import os
import glob
import itertools
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.preprocessing import label_binarize

# Define classes and dictionary
map = ["FCe", "FCmu", "PCe", "PCmu"]
dictio = {v: i for i, v in enumerate(map)}

# Function to load and preprocess images
def load_and_preprocess_image(path, target_size):
    img = load_img(path, target_size=target_size)
    img = img_to_array(img)
    img /= 255.0 # Normalize to [0,1]
    return np.expand_dims(img, axis=0) # Expand dims to make it (1, target_size, target_size, 3)

# Function to predict events and extract true labels
def predict_event(model, directory, target_size=(224, 224)):
    files = glob.glob(os.path.join(directory, '*.png')) # Assuming images are .png
    predictions = []
    true_labels = []
    for file_path in files:
        img = load_and_preprocess_image(file_path, target_size)
        pred = model.predict(img)
        predictions.append(pred[0])
        name = os.path.basename(file_path)[7:-4] # Remove progessive index, underscore and '.png'
        true_labels.append(name)
    return predictions, true_labels

# Load the saved model
model = load_model('best_model.keras')

# Use the model to predict labels
new_data_dir = "event_display_evaluate" # Update this path to new images directory
predictions, true_labels = predict_event(model, new_data_dir)

# Encoding labels
label_encoder = LabelEncoder()
label_encoder.fit(true_labels) # Fit encoder on the actual labels
encoded_labels = label_encoder.transform(true_labels)

# Convert predictions to class indices
pred_classes = np.argmax(predictions, axis=1)

# Generate confusion matrix
cm = confusion_matrix(encoded_labels, pred_classes)
print("\nConfusion Matrix:\n", cm)

# Classification Report
report = classification_report(encoded_labels, pred_classes, target_names=map)
print("\nClassification Report:\n", report)

# F1 Scores
f1_macro = f1_score(encoded_labels, pred_classes, average='macro')
f1_weighted = f1_score(encoded_labels, pred_classes, average='weighted')
print(f"F1 Score (Macro): {f1_macro:.4f}")
print(f"F1 Score (Weighted): {f1_weighted:.4f}")

# Function to plot confusion matrix
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion Matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized Confusion Matrix")
    else:
        print("Confusion Matrix, without normalization")

    plt.figure(figsize=(8,6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()

# Plot non-normalized confusion matrix
plot_confusion_matrix(cm, classes=map, title='Confusion Matrix')

# Plot normalized confusion matrix
plot_confusion_matrix(cm, classes=map, normalize=True, title='Normalized Confusion Matrix')

plt.show()

# Binarize the true labels for multi-class ROC
true_labels_binarized = label_binarize(encoded_labels, classes=range(len(map)))

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(len(map)):
    fpr[i], tpr[i], _ = roc_curve(true_labels_binarized[:, i], np.array(predictions)[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Function to plot ROC curve for two specified classes
def plot_roc_two_classes(preds, labels, class1, class2, class_indices, idx_to_class):
    # Get indices
    class1_idx = class_indices[class1]
    class2_idx = class_indices[class2]

    # Filter the dataset for the two classes
    binary_indices = np.where((labels == class1_idx) | (labels == class2_idx))[0]
    binary_true = labels[binary_indices]
    binary_pred = preds[binary_indices][:, [class1_idx, class2_idx]]

    # Binarize labels: class1=0, class2=1
    binary_true = np.where(binary_true == class1_idx, 0, 1)
    binary_score = binary_pred[:, 1]  # Probability of class2

    # Compute ROC curve and AUC
    fpr_val, tpr_val, _ = roc_curve(binary_true, binary_score)
    roc_auc_val = auc(fpr_val, tpr_val)

    # Plot
    plt.figure(figsize=(6,5))
    plt.plot(fpr_val, tpr_val, label=f"{class1} vs {class2} (AUC = {roc_auc_val:.2f})")
    plt.plot([0, 1], [0, 1], 'r--') # Diagonal
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve: {class1} vs {class2}')
    plt.legend(loc="lower right")
    plt.show()

# Mapping from class index to class name
idx_to_class = {v: k for k, v in dictio.items()}

# Class pairs
class_pairs = [
    ("FCe", "FCmu"),
    ("PCe", "PCmu"),
    ("FCe", "PCe"),
    ("FCmu", "PCmu")
]

# Plotting ROC curve for each pair
for pair in class_pairs:
    plot_roc_two_classes(np.array(predictions), encoded_labels, pair[0], pair[1], dictio, idx_to_class)
