import os
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

# Function to check the current working directory
def check_current_directory():
    print("Current Working Directory:", os.getcwd())

# Function to check the data directory structure
def check_data_directory_structure(data_dir):
    if os.path.exists(data_dir):
        print(f"The directory '{data_dir}' exists.")
        print("Contents of the directory:")
        for item in os.listdir(data_dir):
            print(" -", item)
    else:
        print(f"The directory '{data_dir}' does not exist.")

# Function to check if images are readable
def check_images(generator):
    for i in range(len(generator)):
        x, y = generator[i]
        for img in x:
            if img is None:
                print("Found an unreadable image.")
                return
    print("All images are readable.")

# Function to create a simple model
def create_model():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(7, activation='softmax'))  # Assuming 7 classes for skin types
    return model

# Main debugging function
def main():
    # Check current directory
    check_current_directory()

    # Check data directory structure
    check_data_directory_structure('Data/fairface/fairface/train')
    check_data_directory_structure('Data/fairface/fairface/val')

    # Initialize image data generators
    train_data_gen = ImageDataGenerator(rescale=1./255)
    validation_data_gen = ImageDataGenerator(rescale=1./255)

    # Create data generators
    train_generator = train_data_gen.flow_from_directory(
        'Data/fairface/fairface/train',
        target_size=(48, 48),
        batch_size=64,
        color_mode="grayscale",
        class_mode='categorical'
    )

    validation_generator = validation_data_gen.flow_from_directory(
        'Data/fairface/fairface/val',
        target_size=(48, 48),
        batch_size=64,
        color_mode="grayscale",
        class_mode='categorical'
    )

    # Check if images are readable
    check_images(train_generator)
    check_images(validation_generator)

    # Create and compile the model
    model = create_model()
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.0001), metrics=['accuracy'])

    # Print model summary
    model.summary()

    # Attempt to run a small training loop
    try:
        model.fit(train_generator, steps_per_epoch=10, epochs=1, validation_data=validation_generator, validation_steps=5)
        print("Training completed successfully.")
    except Exception as e:
        print("An error occurred during training:", str(e))

if __name__ == "__main__":
    main()