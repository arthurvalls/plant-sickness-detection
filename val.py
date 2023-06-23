import cv2
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

# Load the saved model
model = load_model('plant.h5')

# Load the image to be tested
img_name = 'planta'
img_path = f'{img_name}.jpg'
img = cv2.imread(img_path)
img = cv2.resize(img, (150, 150))

# Convert the image to a 3D tensor
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)

# Normalize pixel values to [0, 1]
x /= 255.0

# Use the model to predict whether the plant is healthy or sick
prediction = model.predict(x)

# Write the prediction on the image
if prediction < 0.5:
    text = "Healthy"
    color = (0, 255, 0)  # Green
else:
    text = "Sick"
    color = (0, 0, 255)  # Red

cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

# Save the image with the prediction
output_path = f'{img_name}_prediction.jpg'
cv2.imwrite(output_path, img)
print(f"Image saved to {output_path}")
