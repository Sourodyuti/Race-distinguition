from keras.preprocessing.image import ImageDataGenerator
import os
import tensorflow as tf

# Paths
organized_image_folder = "OrganisedDataset/val"
augmented_image_folder = "AugmentedDataset/val"

# Define the ImageDataGenerator with augmentations
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest"
)

# Augment images for each label folder
for label in os.listdir(organized_image_folder):
    label_folder = os.path.join(organized_image_folder, label)
    augmented_label_folder = os.path.join(augmented_image_folder, label)
    os.makedirs(augmented_label_folder, exist_ok=True)

    for img_file in os.listdir(label_folder):
        img_path = os.path.join(label_folder, img_file)
        img = tf.keras.utils.load_img(img_path)  # Load image
        img_array = tf.keras.utils.img_to_array(img)  # Convert to array
        img_array = img_array.reshape((1,) + img_array.shape)  # Reshape for augmentation

        # Save augmented images
        save_prefix = os.path.splitext(img_file)[0]
        i = 0
        for batch in datagen.flow(img_array, batch_size=1, save_to_dir=augmented_label_folder, save_prefix=save_prefix, save_format="jpg"):
            i += 1
            if i > 5:  # Generate 5 augmented images per original image
                break
