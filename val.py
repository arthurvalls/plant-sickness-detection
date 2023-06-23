import tkinter as tk
from tkinter import filedialog
import cv2
import numpy as np
from PIL import ImageTk, Image
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('model/plant.h5')

# Create a GUI window
window = tk.Tk()
window.title("Plant Health Prediction")
window.geometry("500x400")

# Function to handle the image upload
def upload_image():
    # Open a file dialog to select an image file
    file_path = filedialog.askopenfilename()
    if file_path:
        # Load the selected image and resize it to (150,150)
        img = cv2.imread(file_path)
        img = cv2.resize(img, (150, 150))

        # Convert the image to a 3D tensor and normalize the pixel values between 0 and 1
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x /= 255.0

        # Use the loaded model to predict whether the plant is healthy or sick
        prediction = model.predict(x)

        # Display the prediction result on the GUI interface along with the uploaded image
        if prediction < 0.5:
            result_label.config(text="Healthy", fg="green")
        else:
            result_label.config(text="Sick", fg="red")

        # Convert the image to a format that can be displayed on the GUI window
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img = ImageTk.PhotoImage(img)

        # Display the uploaded image on the GUI window
        image_label.img = img
        image_label.config(image=img)

# Create a "Upload" button
upload_button = tk.Button(window, text="Upload", command=upload_image)
upload_button.pack(pady=10)

# Create a label to display the prediction result
result_label = tk.Label(window, font=("Arial", 16))
result_label.pack(pady=10)

# Create a label to display the uploaded image
image_label = tk.Label(window)
image_label.pack(pady=10)

# Create a "Close" button
close_button = tk.Button(window, text="Close", command=window.destroy)
close_button.pack(pady=10)

# Run the GUI window
window.mainloop()
