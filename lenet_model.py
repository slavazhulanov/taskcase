from model.lenet import LeNet
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from keras.optimizers import SGD
import matplotlib.pyplot as plt
from imutils import paths
import numpy as np
import cv2
import os

# получаем пути к изображениям
imagePaths = list(paths.list_images("dataset"))

# подготовненные списки для сохраниния изображений и лейблов
data = []
labels = []

print("[INFO] Загрузка датасета..")
for (i, imagePath) in enumerate(imagePaths):
    image = cv2.imread(imagePath)
    label = imagePath.split(os.path.sep)[-2]

    data.append(image)
    labels.append(label)

# перевод в массивы
data, labels = np.array(data), np.array(labels)

# нормализируем изображения в диапазоне [0, 1]
data = data.astype("float") / 255.0

# разбиваем данные на тренировочную (75%) и тестовую (25%) выборку
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=42)

# преобразуем метки из целых чисел в векторы
trainY = LabelBinarizer().fit_transform(trainY)
testY = LabelBinarizer().fit_transform(testY)

print("[INFO] Компиляция модели..")
opt = SGD(lr=0.01,decay=0.01 / 20, momentum=0.9, nesterov=True)
model = LeNet.build(width=28, height=28, depth=3, classes=10)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

print("[INFO] Тренеровка модели..")
H = model.fit(trainX, trainY, validation_data=(testX, testY), batch_size=32, epochs=15, verbose=1)

print("[INFO] Сохранение модели..")
model.save("saved_model/lenet_model.hdf5")

print("[INFO] Прогнозирование..")
predictions = model.predict(testX, batch_size=32)

print("[INFO] Создание графика потерь и точности на тренировке и валидации..")
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 15), H.history["loss"], label="Потери на тренировке")
plt.plot(np.arange(0, 15), H.history["val_loss"], label="Потери на валидации")
plt.plot(np.arange(0, 15), H.history["accuracy"], label="Точность на тренировке")
plt.plot(np.arange(0, 15), H.history["val_accuracy"], label="Точность на валидации")
plt.title("Потери и точность")
plt.xlabel("Эпохи")
plt.ylabel("Потери/Точность")
plt.legend()
plt.savefig("output_img/lenet_plot.png")