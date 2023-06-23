import os
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.optimizers import Adam

# Set the path to your main dataset folder
dataset_path = 'dataset'

# Define the image dimensions and the batch size
img_width, img_height = 150, 150
batch_size = 8

# Create an ImageDataGenerator for data augmentation and preprocessing
datagen = ImageDataGenerator(
    rescale=1.0 / 255,      # Normalize pixel values to [0, 1]
    shear_range=0.2,        # Apply random shear transformations
    zoom_range=0.2,         # Apply random zoom transformations
    horizontal_flip=True   # Flip images horizontally
)

# Load and prepare the training data
train_generator = datagen.flow_from_directory(
    os.path.join(dataset_path, 'train'),
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary',
    classes=['healthy', 'sick']
)

# Load and prepare the validation data
validation_generator = datagen.flow_from_directory(
    os.path.join(dataset_path, 'validation'),
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary',
    classes=['healthy', 'sick']
)

# Define the neural network architecture
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=3,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size
)

# Save the model
model.save('plant.h5')
