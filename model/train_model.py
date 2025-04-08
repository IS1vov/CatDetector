import os
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers, models

cat_images_dir = "data/cats/"
not_cat_images_dir = "data/non_cats/"


def load_and_preprocess_images(image_dir, label):
    images = []
    labels = []

    for img_name in os.listdir(image_dir):
        img_path = os.path.join(image_dir, img_name)

        if img_name.lower() == "thumbs.db":
            continue

        try:
            img = Image.open(img_path)
            img = img.resize((128, 128))
            img = np.array(img) / 255.0

            if img.shape == (128, 128, 3):
                images.append(img)
                labels.append(label)
        except Exception as e:
            print(f"Ошибка при загрузке изображения {img_name}: {e}")

    return np.array(images), np.array(labels)


cats_images, cats_labels = load_and_preprocess_images(cat_images_dir, 1)
not_cats_images, not_cats_labels = load_and_preprocess_images(not_cat_images_dir, 0)

X = np.concatenate([cats_images, not_cats_images], axis=0)
y = np.concatenate([cats_labels, not_cats_labels], axis=0)

model = models.Sequential(
    [
        layers.Conv2D(32, (3, 3), activation="relu", input_shape=(128, 128, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dense(1, activation="sigmoid"),
    ]
)

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

history = model.fit(X, y, epochs=10, batch_size=32, validation_split=0.2)

model.save("model/model.h5")
