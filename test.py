# import os
# import pandas as pd
# import numpy as np
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing.image import load_img, img_to_array
# from PIL import Image

# # Set image dimensions and parameters
# img_width, img_height = 128, 128

# # Function to resize images and save to a new directory
# def resize_images(input_dir, output_dir, target_size):
#     os.makedirs(output_dir, exist_ok=True)
    
#     for image_name in os.listdir(input_dir):
#         image_path = os.path.join(input_dir, image_name)
        
#         # Skip directories
#         if os.path.isdir(image_path):
#             print(f"Skipping directory: {image_name}")
#             continue
        
#         try:
#             with Image.open(image_path) as img:
#                 img.thumbnail(target_size, Image.LANCZOS)  # Resize while maintaining aspect ratio

#                 # Create a new image with the target size and a white background
#                 new_image = Image.new("RGB", target_size, (255, 255, 255))
#                 x_offset = (target_size[0] - img.width) // 2
#                 y_offset = (target_size[1] - img.height) // 2
#                 new_image.paste(img, (x_offset, y_offset))

#                 # Save the resized image to the output directory
#                 output_image_path = os.path.join(output_dir, image_name)
#                 new_image.save(output_image_path)
#                 print(f"Resized and saved: {output_image_path}")
        
#         except Exception as e:
#             print(f"Error processing image {image_name}: {e}")

# # Step 1: Resize images before loading them into the model
# resize_images('datasets/test/', 'datasets/resized_test/', (img_width, img_height))

# # Step 2: Load the trained model
# model = load_model('cat_dog_classifier.h5')

# # Initialize a list to store image names and predictions
# results = []

# # Step 3: Loop over the resized images and make predictions
# resized_image_dir = 'datasets/resized_test/'
# for image_name in os.listdir(resized_image_dir):
#     image_path = os.path.join(resized_image_dir, image_name)
    
#     # Skip directories
#     if os.path.isdir(image_path):
#         print(f"Skipping directory: {image_name}")
#         continue
    
#     try:
#         # Load the image, preprocess it
#         img = load_img(image_path, target_size=(img_width, img_height))  # Resize image (redundant here, just for safety)
#         img_array = img_to_array(img) / 255.0  # Normalize pixel values to [0, 1]
#         img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        
#         # Predict the class (cat or dog)
#         prediction = model.predict(img_array)[0][0]
#         predicted_class = 'Dog' if prediction > 0.5 else 'Cat'  # Threshold for binary classification
        
#         # Save the result (image name and predicted class)
#         results.append([image_name, predicted_class])
    
#     except Exception as e:
#         print(f"Error processing image {image_name}: {e}")

# # Step 4: Save the results to an Excel file
# df = pd.DataFrame(results, columns=['Image Name', 'Predicted Class'])
# output_file = 'cat_dog_predictions.xlsx'
# df.to_excel(output_file, index=False)

# print(f"Predictions saved to {output_file}")


# #test without performance metrics
import os
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image

# Set image dimensions and parameters
# img_width, img_height = 128, 128
img_width, img_height = 224, 224

# Function to resize images and save to a new directory
def resize_images(input_dir, output_dir, target_size):
    os.makedirs(output_dir, exist_ok=True)
    
    for image_name in os.listdir(input_dir):
        image_path = os.path.join(input_dir, image_name)
        
        # Skip directories
        if os.path.isdir(image_path):
            print(f"Skipping directory: {image_name}")
            continue
        
        try:
            with Image.open(image_path) as img:
                img.thumbnail(target_size, Image.LANCZOS)  # Resize while maintaining aspect ratio

                # Create a new image with the target size and a white background
                new_image = Image.new("RGB", target_size, (255, 255, 255))
                x_offset = (target_size[0] - img.width) // 2
                y_offset = (target_size[1] - img.height) // 2
                new_image.paste(img, (x_offset, y_offset))

                # Save the resized image to the output directory
                output_image_path = os.path.join(output_dir, image_name)
                new_image.save(output_image_path)
                print(f"Resized and saved: {output_image_path}")
        
        except Exception as e:
            print(f"Error processing image {image_name}: {e}")

# Step 1: Resize images before loading them into the model
resize_images('datasets/test/', 'datasets/resized_test/', (img_width, img_height))

# Step 2: Load the trained model
# model = load_model('cat_dog_classifier.h5')
model = load_model('cat_dog_classifier_no_preprocessing.h5')

# Initialize a list to store ids and labels
results = []

# Step 3: Loop over the resized images (sorted) and make predictions
resized_image_dir = 'datasets/resized_test/'
image_list = sorted(os.listdir(resized_image_dir), key=lambda x: int(x.split('.')[0]))  # Sort by numeric part of image names

for image_name in image_list:
    image_path = os.path.join(resized_image_dir, image_name)
    
    # Skip directories
    if os.path.isdir(image_path):
        print(f"Skipping directory: {image_name}")
        continue
    
    try:
        # Load the image, preprocess it
        img = load_img(image_path, target_size=(img_width, img_height))  # Resize image (redundant here, just for safety)
        img_array = img_to_array(img) / 255.0  # Normalize pixel values to [0, 1]
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        
        # Predict the class (cat or dog)
        prediction = model.predict(img_array)[0][0]
        predicted_label = 1 if prediction > 0.5 else 0  # 1 for Dog, 0 for Cat
        
        # Extract the numeric part of the image filename to use as the id
        image_id = int(image_name.split('.')[0])
        
        # Save the result (id and label)
        results.append([image_id, predicted_label])
    
    except Exception as e:
        print(f"Error processing image {image_name}: {e}")

# Step 4: Sort results by id and save to an Excel file with columns 'id' and 'label'
results = sorted(results, key=lambda x: x[0])  # Ensure the results are sorted by id
df = pd.DataFrame(results, columns=['id', 'label'])
output_file = 'cnnwithoutprep_cat_dog_predictions.xlsx'
df.to_excel(output_file, index=False)

print(f"Predictions saved to {output_file}")