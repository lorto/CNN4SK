import warnings
import os
import glob
import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Ignore warnings
warnings.filterwarnings("ignore")

# Check TensorFlow version
print("TensorFlow version:", tf.__version__)

# Check for GPU availability
device_list = tf.config.list_physical_devices('GPU')
print("Available GPU devices:", device_list)

data_dir = "event_display" # Path for folders
classes = ["FCe", "FCmu", "PCe", "PCmu"]

# Count images per class
img_counts = {}
for cls in classes:
    cls_path = os.path.join(data_dir, cls)
    images_in_cls = glob.glob(os.path.join(cls_path, "*.png"))
    img_counts[cls] = len(images_in_cls)

print("Image counts per class:")
for k, v in img_counts.items():
    print(f"   {k}: {v} images")

# Display a sample image for each class
plt.figure(figsize=(12, 3))
for i, cls in enumerate(classes):
    cls_path = os.path.join(data_dir, cls)
    images_in_cls = glob.glob(os.path.join(cls_path, "*.png"))
    if len(images_in_cls) > 0:
        img_path = images_in_cls[0] # Take the first image
        image_data = mpimg.imread(img_path)
        plt.subplot(1, 4, i+1)
        plt.imshow(image_data, cmap='gray')
        plt.title(f"Sample: {cls}")
        plt.axis('off')
plt.tight_layout()
plt.show()

# Define batch size and target image size
batch_size = 16
img_height = 224 # ResNet50 expects at least 197x197
img_width = 224

# ImageDataGenerator for training with augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.3, # 30% for validation
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Training generator
train_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

# Validation generator
validation_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

# Display class indices
print("Class indices:", train_generator.class_indices)

# Load ResNet50 with pre-trained ImageNet weights, excluding the top classification layers
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))

# Freeze the base_model to prevent its weights from being updated during training
base_model.trainable = False

# Adding custom layers on top of ResNet50
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(len(classes), activation='softmax')(x)

# Define the complete model
model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
learning_rate = 1e-4
model.compile(
    optimizer=Adam(learning_rate=learning_rate),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Display model summary
model.summary()

# Define the number of epochs, steps per epoch and validation steps
epochs = 25
steps_per_epoch = math.ceil(train_generator.samples / batch_size)
validation_steps = math.ceil(validation_generator.samples / batch_size)

# Define callbacks
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
checkpoint = ModelCheckpoint('best_model.keras', monitor='val_loss', save_best_only=True)

# Train the model
history = model.fit(
    train_generator,
    epochs=epochs,
    steps_per_epoch=steps_per_epoch,
    validation_data=validation_generator,
    validation_steps=validation_steps,
    callbacks=[early_stop, checkpoint]
)
