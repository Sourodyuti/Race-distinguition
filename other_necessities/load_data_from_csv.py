import os
import pandas as pd
import numpy as np
import cv2
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from keras.optimizers import Adam
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Function to load images and labels from a CSV file
def load_data_from_csv(csv_file):
    data = pd.read_csv(csv_file)
    images = []
    labels = []
    
    for index, row in data.iterrows():
        img_path = row['image_path']
        label = row['label']
        
        # Load the image
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, (48, 48))  # Resize to match model input
            images.append(img)
            labels.append(label)
        else:
            print(f"Warning: Image at {img_path} could not be loaded.")
    
    return np.array(images), np.array(labels)

# Function to create a simple model
def create_model(num_classes):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))  # Adjust for number of classes
    return model

# Main function
def main():
    # Load data from CSV
    images, labels = load_data_from_csv('UnorganisedDataset/fitz_undersampled_test_final.csv')  # Update with your CSV file path

    # Convert labels to categorical
    unique_labels = np.unique(labels)
    label_to_index = {label: index for index, label in enumerate(unique_labels)}
    indexed_labels = [label_to_index[label] for label in labels]
    indexed_labels = to_categorical(indexed_labels, num_classes=len(unique_labels))

    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(images, indexed_labels, test_size=0.2, random_state=42)

    # Reshape images for the model
    X_train = X_train.reshape(-1, 48, 48, 1).astype('float32') / 255.0
    X_val = X_val.reshape(-1, 48, 48, 1).astype('float32') / 255.0

    # Create and compile the model
    model = create_model(num_classes=len(unique_labels))
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.0001), metrics=['accuracy'])

    # Train the model
    model.fit(X_train, y_train, epochs=50, batch_size=64, validation_data=(X_val, y_val))

    # Save the model
    model.save('skin_type_model.h5')

if __name__ == "__main__":
    main()