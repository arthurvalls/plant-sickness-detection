import os
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.optimizers import Adam

# Define o caminho para a pasta principal do conjunto de dados
dataset_path = 'dataset'

# Define as dimensões da imagem e o tamanho do batch
img_width, img_height = 150, 150
batch_size = 8

# Cria um ImageDataGenerator para aumentar e pré-processar os dados
datagen = ImageDataGenerator(
    rescale=1.0 / 255,      # Normaliza os valores dos pixels entre 0 e 1
    shear_range=0.2,        # Aplica transformações de cisalhamento aleatórias
    zoom_range=0.2,         # Aplica transformações de zoom aleatórias
    horizontal_flip=True   # Inverte as imagens horizontalmente
)

# Carrega e prepara os dados de treinamento
train_generator = datagen.flow_from_directory(
    os.path.join(dataset_path, 'train'),
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary',
    classes=['healthy', 'sick']
)

# Carrega e prepara os dados de validação
validation_generator = datagen.flow_from_directory(
    os.path.join(dataset_path, 'validation'),
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary',
    classes=['healthy', 'sick']
)

# Define a arquitetura da rede neural
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compila o modelo
model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# Treina o modelo
model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=3,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size
)

# Salva o modelo
model.save('model/plant.h5')
