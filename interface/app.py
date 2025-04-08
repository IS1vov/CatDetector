import tkinter as tk
from tkinter import filedialog, messagebox
from tensorflow.keras.models import load_model
from PIL import Image, ImageTk
import pygame
import numpy as np

# Загрузка модели
model = load_model("model/model.h5")

# Инициализация pygame для звуков
pygame.mixer.init()


# Функция для воспроизведения звуков
def play_sound(sound_file):
    pygame.mixer.music.load(sound_file)
    pygame.mixer.music.play()


# Функция для предсказания на изображении
def predict_image(img_path):
    img = Image.open(img_path)
    img = img.convert("RGB")
    img = img.resize((128, 128))  # изменяем размер изображения
    img = np.array(img) / 255.0  # нормализуем изображение
    img = np.expand_dims(img, axis=0)  # добавляем размерность для батча

    # Получаем предсказание
    prediction = model.predict(img)

    if prediction >= 0.5:
        return "КОТИК ОБНАРУЖЕН", "data/meow.mp3"
    else:
        return "КОТИК НЕ ОБНАРУЖЕН", "data/nonmeow.mp3"


# Функция для обработки выбора файла
def upload_image():
    img_path = filedialog.askopenfilename(
        title="Выберите изображение", filetypes=[("Image files", "*.jpg *.jpeg *.png")]
    )
    if img_path:
        label_result.config(text="Загрузка...", fg="white")
        result_text, sound_file = predict_image(img_path)
        label_result.config(
            text=result_text,
            font=("Arial", 40),
            fg="green" if "КОТИК ОБНАРУЖЕН" in result_text else "red",
        )
        play_sound(sound_file)

        # Отображение выбранного изображения
        img = Image.open(img_path)
        img = img.resize((250, 250))  # изменяем размер изображения
        img = ImageTk.PhotoImage(img)
        panel_img.config(image=img)
        panel_img.image = img


# Создание основного окна
root = tk.Tk()
root.title("Ищем Котиков")
root.geometry("600x600")
root.configure(bg="black")

# Логотип (белыми буквами)
label_logo = tk.Label(
    root, text="Ищем Котиков", font=("Arial", 40), fg="white", bg="black"
)
label_logo.pack(pady=50)

# Кнопка для загрузки изображения
button_upload = tk.Button(
    root, text="Загрузить изображение", font=("Arial", 20), command=upload_image
)
button_upload.pack(pady=20)

# Метка для результата
label_result = tk.Label(
    root, text="Результат", font=("Arial", 40), fg="white", bg="black"
)
label_result.pack(pady=20)

# Панель для отображения изображения
panel_img = tk.Label(root)
panel_img.pack(pady=20)

# Запуск главного цикла приложения
root.mainloop()
