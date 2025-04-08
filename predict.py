import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from PIL import Image

# Загрузка обученной модели
model = load_model("model/model.h5")


# Функция для предсказания на изображении
def predict_image(img_path):
    img = Image.open(img_path)
    img = img.resize((128, 128))  # изменяем размер изображения
    img = np.array(img) / 255.0  # нормализуем изображение

    # Добавляем размерность для батча
    img = np.expand_dims(img, axis=0)

    # Получаем предсказание
    prediction = model.predict(img)

    if prediction >= 0.5:
        return "КОТИК ОБНАРУЖЕН"
    else:
        return "КОТИК НЕ ОБНАРУЖЕН"


# Визуализация графиков точности
def plot_accuracy(history):
    plt.plot(history.history["accuracy"], label="Точность на обучении")
    plt.plot(history.history["val_accuracy"], label="Точность на валидации")
    plt.xlabel("Эпохи")
    plt.ylabel("Точность")
    plt.legend()
    plt.show()

    plt.plot(history.history["loss"], label="Потери на обучении")
    plt.plot(history.history["val_loss"], label="Потери на валидации")
    plt.xlabel("Эпохи")
    plt.ylabel("Потери")
    plt.legend()
    plt.show()


# Загрузка тестового изображения и предсказание
img_path = "/Users/is/Desktop/catdetector/black-adult-house-cat-standing-600nw-2184634675.jpg"  # Укажи путь к своему тестовому изображению
result = predict_image(img_path)
print(result)

# Загрузка истории обучения (если необходимо)
history = None
# Если у тебя есть история из предыдущего обучения, можно загрузить её
# Например, если у тебя была сохранена история в файле, загрузите её тут

# Отображение графиков точности, если есть история
if history is not None:
    plot_accuracy(history)
