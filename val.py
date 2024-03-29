import tkinter as tk
from tkinter import filedialog
import cv2
import numpy as np
from PIL import ImageTk, Image
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

# Carrega o modelo treinado
model = load_model('model/plant_best_50_epochs.h5')

# Cria a interface
window = tk.Tk()
window.title("Previsão de Plantas Doentes")
window.geometry("700x700")

# Função para realizar upload de imagem
def upload_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        img = cv2.imread(file_path)
        img = cv2.resize(img, (400, 400))

        # Converte a imagem para um tensor 3D e normaliza os pixels entre 0 e 1 
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x /= 255.0

        # Realiza a previsão se a planta está saudável e doente
        prediction = model.predict(x)
        print(prediction)

        # Mostra o resultado da previsão da imagem
        if prediction < 0.5:
            result_label.config(text=f"Saudável: {100*(1-prediction[0][0]):.2f}%", fg="green")
        else:
            result_label.config(text=f"Doente: {(100*prediction[0][0]):.2f}%", fg="red")

        # Converte e mostra a imagem 
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img = ImageTk.PhotoImage(img)
        image_label.img = img
        image_label.config(image=img)

# Cria o butao de upload
upload_button = tk.Button(window, text="Teste uma imagem", command=upload_image)
upload_button.pack(pady=10)

# Mostra o resultado da previsão na interface
result_label = tk.Label(window, font=("Arial", 26))
result_label.pack(pady=10)

# Mostra a imagem testada 
image_label = tk.Label(window)
image_label.pack(pady=10)

# Cria o butão para fechar a interface e o programa
close_button = tk.Button(window, text="Fechar", command=window.destroy)
close_button.pack(pady=10)

# Roda a inteface
window.mainloop()
