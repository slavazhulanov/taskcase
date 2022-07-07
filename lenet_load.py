from keras.models import load_model
from imutils import paths
import numpy as np
import cv2
import os

# инициализируем лейблы
classLabels = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

# получаем пути к изображениям
imagePaths = np.array(list(paths.list_images("dataset")))

# выбираем случайные 10 изображений для дальнейшего дополнения
idxs = np.random.randint(0, len(imagePaths), size=(10,))
imagePaths = imagePaths[idxs]

# подготовненные списки для сохраниния изображений и лейблов
data = []
labels = []

print("[INFO] Загрузка примеров изображений..")
for (i, imagePath) in enumerate(imagePaths):
    image = cv2.imread(imagePath)
    label = imagePath.split(os.path.sep)[-2]

    data.append(image)
    labels.append(label)

# перевод в массивы
data, labels = np.array(data), np.array(labels)

# нормализируем изображения в диапазоне [0, 1]
data = data.astype("float") / 255.0

print("[INFO] Загрузка сохраненной модели..")
model = load_model("saved_model/lenet_model.hdf5")

print("[INFO] Прогнозирование..")
preds = model.predict(data, batch_size=32).argmax(axis=1)

# Цикл преобразования изображений
for (i, imagePath) in enumerate(imagePaths):
    image = cv2.imread(imagePath)
    image = cv2.resize(image, (360, 360)).astype(np.uint8)
    cv2.putText(image, "Label: {}".format(classLabels[preds[i]]),
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow("Image", image)
    cv2.waitKey(0)