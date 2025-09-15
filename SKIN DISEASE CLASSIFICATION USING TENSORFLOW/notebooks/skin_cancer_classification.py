# ======================== LIBRARY IMPORTS ========================
import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import multiprocessing
import concurrent.futures

'''import tensorflow as tf
from tensorflow.keras.applications import DenseNet201
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split'''

# ======================== STEP 1: BUILD DATAFRAME ========================
def create_image_dataframe(directory):
    """Returns a DataFrame with image paths and associated numeric labels"""
    image_records = [
        {"filepath": os.path.join(directory, class_name, file_name), "label": label_id}
        for label_id, class_name in enumerate(os.listdir(directory))
        for file_name in os.listdir(os.path.join(directory, class_name))
    ]
    return pd.DataFrame(image_records)

# Dataset paths
train_folder = 'C:/Users/ben4s/Downloads/Skin cancer ISIC The International Skin Imaging Collaboration/Train'
test_folder =  'C:/Users/ben4s/Downloads/Skin cancer ISIC The International Skin Imaging Collaboration/Test'

# Combine training and testing image info
image_df = pd.concat([create_image_dataframe(train_folder), create_image_dataframe(test_folder)], ignore_index=True)

# Label map: numeric -> class name
label_dictionary = {i: name for i, name in enumerate(os.listdir(train_folder))}
num_classes = len(label_dictionary)

# Balance each class to a max of 2000 images
max_per_class = 2000
image_df = image_df.groupby("label").head(max_per_class).reset_index(drop=True)

# ======================== STEP 2: ENABLE GPU MEMORY GROWTH ========================
try:
    for g in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(g, True)
except:
    pass

# ======================== STEP 3: RESIZE IMAGES ========================
def resize_image(path):
    return np.asarray(Image.open(path).resize((100, 75)))

cpu_cores = multiprocessing.cpu_count()

with concurrent.futures.ThreadPoolExecutor(max_workers=cpu_cores) as executor:
    image_df['array'] = list(executor.map(resize_image, image_df['filepath']))

# ======================== STEP 4: AUGMENT DATA ========================
augmenter = ImageDataGenerator(
    rotation_range=25,
    width_shift_range=0.5,
    height_shift_range=0.25,
    shear_range=0.25,
    zoom_range=0.25,
    horizontal_flip=True,
    fill_mode='nearest'
)

aug_df = pd.DataFrame(columns=['filepath', 'label', 'array'])

for class_val in image_df['label'].unique():
    class_subset = image_df[image_df['label'] == class_val]
    arrays = class_subset['array'].values
    needed_count = max_per_class - len(arrays)

    aug_df = pd.concat([aug_df, class_subset], ignore_index=True)

    if needed_count > 0:
        new_samples = np.random.choice(arrays, size=needed_count, replace=True)
        for sample in new_samples:
            augmented = augmenter.flow(np.expand_dims(sample, axis=0), batch_size=1)
            aug_image = next(augmented)[0].astype('uint8')
            new_row = pd.DataFrame([{'filepath': None, 'label': class_val, 'array': aug_image}])
            aug_df = pd.concat([aug_df, new_row], ignore_index=True)

# Shuffle and cap per class
final_df = aug_df.groupby('label').head(max_per_class).sample(frac=1, random_state=42).reset_index(drop=True)

# ======================== STEP 5: TRAIN-TEST SPLIT ========================
features = final_df['array'].tolist()
targets = final_df['label']

X_train, X_temp, y_train, y_temp = train_test_split(features, targets, test_size=0.2, shuffle=True)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, shuffle=True)

X_train, X_val, X_test = map(np.asarray, [X_train, X_val, X_test])

# Normalize image data
X_train = (X_train - X_train.mean()) / X_train.std()
X_val = (X_val - X_val.mean()) / X_val.std()
X_test = (X_test - X_test.mean()) / X_test.std()

# One-hot encode labels
y_train = to_categorical(y_train, num_classes=num_classes)
y_val = to_categorical(y_val, num_classes=num_classes)
y_test = to_categorical(y_test, num_classes=num_classes)

# Reshape for CNN input
X_train = X_train.reshape(-1, 75, 100, 3)
X_val = X_val.reshape(-1, 75, 100, 3)
X_test = X_test.reshape(-1, 75, 100, 3)

# ======================== STEP 6: BUILD MODEL ========================
input_shape = (75, 100, 3)

cnn_model = Sequential([
    DenseNet201(include_top=False, weights='imagenet', input_shape=input_shape),
    Flatten(),
    Dropout(0.5),
    Dense(512, activation='relu'),
    Dense(num_classes, activation='softmax')
])

cnn_model.summary()

cnn_model.compile(
    optimizer=SGD(learning_rate=0.001, momentum=0.9),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

lr_scheduler = ReduceLROnPlateau(
    monitor='val_accuracy',
    patience=3,
    verbose=1,
    factor=0.5,
    min_lr=1e-5
)

# ======================== STEP 7: TRAIN MODEL ========================
from tensorflow.keras.callbacks import ModelCheckpoint

# Checkpoint to save model whenever validation accuracy improves
checkpoint_callback = ModelCheckpoint(
    filepath='/kaggle/working/best_model.h5',
    monitor='val_accuracy',
    save_best_only=True,
    save_weights_only=False,
    verbose=1
)

# Train the model
history = cnn_model.fit(
    X_train, y_train,
    epochs=30,
    batch_size=32,
    validation_data=(X_val, y_val),
    callbacks=[lr_scheduler, checkpoint_callback]
)

# ======================== STEP 8: SAVE MODEL ========================
cnn_model.save('skin_disease_model.h5')

