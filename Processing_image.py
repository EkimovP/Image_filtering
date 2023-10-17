import numpy as np
import cv2


# Дополнение изображения до степени двойки
class Processing_Image:
    # Интерполяция изображения до степени двойки
    @staticmethod
    def interpolate_to_power_of_two(image):
        # Получаем исходные размеры изображения
        height, width = image.shape
        # Находим ближайшие степени двойки для высоты и ширины
        new_height = 2 ** int(np.ceil(np.log2(height)))
        new_width = 2 ** int(np.ceil(np.log2(width)))

        # Используем интерполяцию для изменения размера изображения до ближайших степеней двойки
        image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

        return image

    # Делаем изображение степени двойки (дополняем нулями и изображение ставим по середине)
    @staticmethod
    def resize_to_power_of_two(original_picture):
        # Получаем исходные размеры изображения
        height, width = original_picture.shape
        # Определяем ближайшие степени двойки для высоты и ширины
        new_height = 2 ** int(np.ceil(np.log2(height)))
        new_width = 2 ** int(np.ceil(np.log2(width)))
        # Рассчитываем смещение, чтобы поместить изображение по середине
        y_offset = (new_height - height) // 2
        x_offset = (new_width - width) // 2

        # Создаем новое изображение заданных размеров и заполняем его нулевыми пикселями
        upscaled_image = np.zeros((new_height, new_width))
        # Срез берём по х, у, по z берем всю и это будет наше исходное изображение
        upscaled_image[y_offset:y_offset + height, x_offset:x_offset + width] = original_picture

        return upscaled_image
