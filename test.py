import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image_dataset_from_directory
import matplotlib.pyplot as plt
import numpy as np

model = load_model('datesModelv2.h5')

dataset = image_dataset_from_directory(
    'dates data',
    image_size=(256, 256),  
    batch_size=1,  
    label_mode='int',
    shuffle=True
)

class_names = dataset.class_names

for images, labels in dataset:
    for i in range(len(images)):
        img = images[i].numpy().astype("uint8")
        label = labels[i].numpy()
        
        prediction = model.predict(tf.expand_dims(images[i], axis=0))

        plt.imshow(img)
        plt.title(f"True: {class_names[label]}, Predicted: {prediction}")
        plt.axis('off')
        plt.show()
