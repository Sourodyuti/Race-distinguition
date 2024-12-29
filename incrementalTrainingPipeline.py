import os
from keras.applications import MobileNetV2
from keras.models import Sequential
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint

# Paths
train_dir = "AugmentedDataset\\train"
val_dir = "AugmentedDataset\\val"

# Image parameters
img_height, img_width = 224, 224
batch_size = 32
num_classes = len(os.listdir(train_dir))  # Automatically detect number of classes

# Load MobileNetV2 base model
base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(img_height, img_width, 3))

# Freeze base layers
for layer in base_model.layers:
    layer.trainable = False

# Add custom layers
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(1024, activation="relu"),
    Dropout(0.5),
    Dense(num_classes, activation="softmax")
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001), loss="categorical_crossentropy", metrics=["accuracy"])

# Data augmentation and preprocessing
train_datagen = ImageDataGenerator(rescale=1.0/255)
val_datagen = ImageDataGenerator(rescale=1.0/255)

train_generator = train_datagen.flow_from_directory(train_dir, target_size=(img_height, img_width), batch_size=batch_size, class_mode="categorical")
val_generator = val_datagen.flow_from_directory(val_dir, target_size=(img_height, img_width), batch_size=batch_size, class_mode="categorical")

# Save checkpoints
checkpoint_path = "checkpoints/skin_tone_model_mobilenetv2_epoch_{epoch:02d}.h5"
checkpoint = ModelCheckpoint(filepath=checkpoint_path, save_best_only=False, save_weights_only=False, verbose=1)

# Train the model incrementally
epochs_per_iteration = 1  # Number of epochs per training session
total_epochs = 20  # Total epochs you want to train
for start_epoch in range(0, total_epochs, epochs_per_iteration):
    print(f"Training epochs {start_epoch+1} to {start_epoch+epochs_per_iteration}")
    model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=start_epoch + epochs_per_iteration,
        initial_epoch=start_epoch,  # Resume from last epoch
        steps_per_epoch=train_generator.samples // batch_size,
        validation_steps=val_generator.samples // batch_size,
        callbacks=[checkpoint]
    )
