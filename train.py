import os
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.optimizers import Adam
import visualkeras

# Define o caminho da pasta principal do conjunto de dados.
dataset_path = '../../dataset'

# Define as dimensões da imagem e o tamanho do lote.
img_width, img_height = 400, 400
batch_size = 16

# Cria um ImageDataGenerator para aumentar e pré-processar os dados.
datagen = ImageDataGenerator(
    rescale=1.0 / 255,  # reescala os pixels da imagem para valores entre 0 e 1.
    shear_range=0.2,  # aplica transformação de cisalhamento nas imagens.
    zoom_range=0.2,  # aplica transformação de zoom nas imagens.
    rotation_range=20,  # aplica transformação de rotação nas imagens.
    width_shift_range=0.2,  # aplica translação horizontal nas imagens.
    height_shift_range=0.2,  # aplica translação vertical nas imagens.
    horizontal_flip=True  # inverte aleatoriamente as imagens horizontalmente.
)


# Carrega e prepara os dados de treinamento.
train_generator = datagen.flow_from_directory(
    os.path.join(dataset_path, 'train'),
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary',
    classes=['healthy', 'sick']
)

# Carrega e prepara os dados de validação.
validation_generator = datagen.flow_from_directory(
    os.path.join(dataset_path, 'test'),
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary',
    classes=['healthy', 'sick']
)

# Define a arquitetura da rede neural.
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compila o modelo.
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Salva o modelo para visualizacao
model.summary()

visualkeras.layered_view(model, legend=True).save('model.png')

# Treina o modelo.
model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=50,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size
)

# Salva o modelo.
model.save('model/plant_best_50_epochs.h5')