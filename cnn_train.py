import os
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import optimizers

# Set image dimensions and parameters
img_width, img_height = 224, 224  # Resize images to 224x224
batch_size = 32
epochs = 50

def resize_images(input_dir, output_dir, target_size):
    os.makedirs(output_dir, exist_ok=True)

    for class_name in os.listdir(input_dir):
        class_dir = os.path.join(input_dir, class_name)
        output_class_dir = os.path.join(output_dir, class_name)
        os.makedirs(output_class_dir, exist_ok=True)  # Create class directory in output

        for image_name in os.listdir(class_dir):
            image_path = os.path.join(class_dir, image_name)
            
            try:
                with Image.open(image_path) as img:
                    img.thumbnail(target_size, Image.LANCZOS)  # Resize while maintaining aspect ratio

                    # Create a new image with the target size and a white background
                    new_image = Image.new("RGB", target_size, (255, 255, 255))
                    x_offset = (target_size[0] - img.width) // 2
                    y_offset = (target_size[1] - img.height) // 2
                    new_image.paste(img, (x_offset, y_offset))

                    # Save the resized image
                    new_image.save(os.path.join(output_class_dir, image_name))
                    print(f"Resized and saved: {image_name} to {output_class_dir}")
            except Exception as e:
                print(f"Error processing image {image_name}: {e}")

# Resize images before loading them into the model
resize_images('datasets/train/', 'datasets/resized_train/', (img_width, img_height))
resize_images('datasets/val/', 'datasets/resized_validation/', (img_width, img_height))

# Data augmentation for training set
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Data preprocessing for validation set
validation_datagen = ImageDataGenerator(rescale=1.0 / 255)

# Load training and validation datasets from resized images
train_set = train_datagen.flow_from_directory(
    'datasets/resized_train/',  # Use the resized images directory
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary'
)

validation_set = validation_datagen.flow_from_directory(
    'datasets/resized_validation/',  # Use the resized validation images directory
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary'
)

# Define the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Compile the model with an appropriate optimizer and loss function
optimizer = optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# Set up Early Stopping and Learning Rate Reduction on Plateau
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6)

# Train the model
history = model.fit(
    train_set,
    validation_data=validation_set,
    epochs=epochs,
    callbacks=[early_stopping, lr_scheduler]
)

# Save the model
model.save('final_cat_dog_classifier.h5')
print("Model saved as final_cat_dog_classifier.h5")

# Plot training & validation accuracy and loss
def plot_metrics(history):
    # Plot Accuracy
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plot Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Save the plots as jpg
    plt.savefig('training_validation_metrics.jpg')
    plt.show()

# Call the function to plot and save the metrics
plot_metrics(history)
