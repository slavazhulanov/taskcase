from keras.preprocessing.image import ImageDataGenerator
from keras.utils import img_to_array
import cv2
import numpy as np

# цикл, который проходит по всем (не жирным) изображениям, генерирует новые и сохраняет их в соответствующие папки

print("[INFO] Генерация изображений с обычным шрифтом..")
for i in range(0, 10):
    input_image_path = "original_images/{}.png".format(i)
    output_dir_path = "dataset/{}".format(i)
    prefix = "{}".format(i)

    image = cv2.imread(input_image_path)
    image = cv2.resize(image, (28, 28)).astype(np.uint8)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)

    # генерация новых изображений
    aug = ImageDataGenerator(rotation_range=30,
                             width_shift_range=0.1,
                             height_shift_range=0.1,
                             shear_range=0.2,
                             zoom_range=0.2,
                             fill_mode="nearest")

    total = 0 #счетчик созданных изображений

    # сохранение данных
    imageGen = aug.flow(image,
                        batch_size=1,
                        save_to_dir=output_dir_path,
                        save_prefix=prefix,
                        save_format="png")

    # генерируем изображения пока счетчик не досчитает до 1000
    for image in imageGen:
         total += 1
         if total == 100:
              break

# цикл, который проходит по всем жирным изображениям, генерирует новые и сохраняет их в соответствующие папки

print("[INFO] Генерация изображений с жирным шрифтом..")
for i in range(0, 10):
    input_image_path = "original_images/{}_thick.png".format(i)
    output_dir_path = "dataset/{}".format(i)
    prefix = "{}".format(i)

    image = cv2.imread(input_image_path, 0)
    image = cv2.resize(image, (28, 28)).astype(np.uint8)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)

    # генерация новых изображений
    aug = ImageDataGenerator(rotation_range=30,
                             width_shift_range=0.1,
                             height_shift_range=0.1,
                             shear_range=0.2,
                             zoom_range=0.2,
                             fill_mode="nearest")

    total = 0 #счетчик созданных изображений

    # сохранение данных
    imageGen = aug.flow(image,
                        batch_size=1,
                        save_to_dir=output_dir_path,
                        save_prefix=prefix,
                        save_format="png")

    # генерируем изображения пока счетчик не досчитает до 1000
    for image in imageGen:
         total += 1
         if total == 100:
              break