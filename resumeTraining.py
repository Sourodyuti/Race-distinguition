import os
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint

# Paths
train_dir = "AugmentedDataset/train"
val_dir = "AugmentedDataset/val"
checkpoint_dir = "checkpoints"

# Image parameters
img_height, img_width = 224, 224
batch_size = 32

# Data augmentation and preprocessing
train_datagen = ImageDataGenerator(rescale=1.0/255)
val_datagen = ImageDataGenerator(rescale=1.0/255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode="categorical"
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode="categorical"
)

# Find the latest checkpoint
checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith(".h5")]
if not checkpoint_files:
    print("No checkpoints found. Starting training from scratch.")
    # Add model creation code here if starting from scratch
else:
    # Sort and load the latest checkpoint
    checkpoint_files.sort()
    latest_checkpoint = os.path.join(checkpoint_dir, checkpoint_files[-1])
    print(f"Loading latest checkpoint: {latest_checkpoint}")
    model = load_model(latest_checkpoint)

# Define the checkpoint callback
checkpoint_callback = ModelCheckpoint(
    filepath=os.path.join(checkpoint_dir, "skin_tone_model_mobilenetv2_epoch_{epoch:02d}.h5"),
    save_best_only=False,
    save_weights_only=False
)

# Resume training
model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=1,  # Continue training for additional epochs
    steps_per_epoch=train_generator.samples // batch_size,
    validation_steps=val_generator.samples // batch_size,
    callbacks=[checkpoint_callback]
)
