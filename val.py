import argparse
import cv2
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

# Pega o path da imagem via linha de comando
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--image', required=True, help='Name of the image file to be tested (without extension)')
args = parser.parse_args()

# Carrega o modelo treinado
model = load_model('model/plant.h5')

# Carrega a imagem a ser testada
img_path = args.image
img = cv2.imread(img_path)
img = cv2.resize(img, (150, 150))

# Converte a imagem ao um tensor 3D 
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)

# Normaliza os valores dos pixels entre 0 e 1
x /= 255.0

# Previsão se a planta está saudável ou doente
prediction = model.predict(x)

# Escreve na imagem a previsão (healthy ou sick)
if prediction < 0.5:
    text = "Healthy"
    color = (0, 255, 0)  # Verde
else:
    text = "Sick"
    color = (0, 0, 255)  # Vermelho

cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

# Salva a imagem com a previsão
output_path = f'assets/{args.image.split(".")[0]}_prediction.jpg'
cv2.imwrite(output_path, img)
print(f"Image salva em {output_path}")
