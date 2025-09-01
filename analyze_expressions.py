import os
import cv2
import matplotlib.pyplot as plt
from deepface import DeepFace

# Set path to your folder
folder_path = "facial expression"

# Get list of image files
image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

# Loop through each image file
for image_name in image_files:
    image_path = os.path.join(folder_path, image_name)
    
    print(f"\nProcessing: {image_name}")
    
    # Read image
    img = cv2.imread(image_path)

    # Show image
    plt.imshow(img[:, :, ::-1])
    
    plt.title(image_name)
    plt.axis('off')
    plt.show()

    # Analyze image
    try:
        result = DeepFace.analyze(img, actions=['emotion'])
        dominant_emotion = result[0]['dominant_emotion']
        print(f"Dominant Emotion: {dominant_emotion}")
    except Exception as e:
        print(f"Could not process {image_name}. Error: {e}")
