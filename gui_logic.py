from gui import Ui_Dialog
from graph import Graph
from drawer import Drawer as drawer
from FastFourierTransform import FFT
from Processing_image import Processing_Image

from PyQt5.QtWidgets import QFileDialog
from tkinter import *
from PyQt5 import QtCore
import random
import cv2
import math
import numpy as np
import matplotlib

matplotlib.use('TkAgg')


# Гауссов купол
def dome_2d(amplitude, x, x_0, sigma_x, y, y_0, sigma_y):
    return amplitude * math.exp(-(
            (
                    ((x - x_0) * (x - x_0)) /
                    (2 * sigma_x * sigma_x)
            ) + (
                    ((y - y_0) * (y - y_0)) /
                    (2 * sigma_y * sigma_y)
            )))


# Перевод цветного изображения в серое
def black_white_image(color_picture):
    height, width, _ = color_picture.shape
    gray_image = np.zeros((height, width), dtype=int)
    for i in range(height):
        for j in range(width):
            pixel = color_picture[i, j]
            gray_image[i, j] = 0.299 * pixel[0] + 0.587 * pixel[1] + 0.114 * pixel[2]
    maximum_intensity = np.max(gray_image)
    multiplier = 255 / maximum_intensity
    gray_image = gray_image * multiplier
    return gray_image


# Окно для предупреждения, если изображение не является степенью двойки
def window():
    root = Tk()
    root.title("Предупреждение!")
    label = Label(root, text='Изображение НЕ является степенью двойки.', fg='black')
    label.pack()
    root.geometry("270x50+670+250")  # расположение окна на экране
    root.mainloop()


# Функция для шума (нормальное распределение по Гауссу)
def uniform_distribution():
    repeat = 12
    val = 0
    for i in range(repeat):
        val += random.random()  # значение от 0.0 до 1.0
    return val / repeat


# Функция, которая возвращает энергию изображения
def energy_pictures(pictures):
    energy = 0
    for picture_line in pictures:
        for pixel in picture_line:
            energy += pixel * pixel
    return energy


# КЛАСС АЛГОРИТМА ПРИЛОЖЕНИЯ
class GuiProgram(Ui_Dialog):

    def __init__(self, dialog):
        # Создаем окно
        Ui_Dialog.__init__(self)
        # Дополнительные функции окна
        dialog.setWindowFlags(  # Передаем флаги создания окна
            QtCore.Qt.WindowCloseButtonHint |  # Закрытие
            QtCore.Qt.WindowMaximizeButtonHint |  # Во весь экран (развернуть)
            QtCore.Qt.WindowMinimizeButtonHint  # Свернуть
        )
        self.setupUi(dialog)  # Устанавливаем пользовательский интерфейс
        # ПОЛЯ КЛАССА
        # Параметры 1 графика - Исходное изображение
        self.graph_1 = Graph(
            layout=self.layout_plot,
            widget=self.widget_plot,
            name_graphics="График №1. Исходное изображение"
        )
        # Параметры 2 графика - Исходное изображение с шумом
        self.graph_2 = Graph(
            layout=self.layout_plot_2,
            widget=self.widget_plot_2,
            name_graphics="График №2. Исходное изображение c шумом"
        )
        # Параметры 3 графика - Спектр, со сменой по диагонали и зоной процента энергии
        self.graph_3 = Graph(
            layout=self.layout_plot_3,
            widget=self.widget_plot_3,
            name_graphics="График №3. Спектр изображения"
        )
        # Параметры 4 графика - Восстановленное изображение
        self.graph_4 = Graph(
            layout=self.layout_plot_4,
            widget=self.widget_plot_4,
            name_graphics="График №4. Отфильтрованное изображение"
        )

        # Картинки этапов обработки
        self.color_picture = None
        self.original_picture = None
        self.gray_picture = None
        self.noise_image = None
        self.picture_spectrum = None
        self.module_spectrum_repositioned = None
        self.half_rectangle_width = None
        self.half_rectangle_height = None
        self.energy_boundaries_x = None
        self.energy_boundaries_y = None
        self.module_reconstructed_image = None

        # ДЕЙСТВИЯ ПРИ ВКЛЮЧЕНИИ
        # Смена режима отображения изображения
        self.radioButton_color_picture.clicked.connect(self.change_picture_to_colored)
        self.radioButton_gray_picture.clicked.connect(self.change_picture_to_gray)
        # Выбрано изображение или график
        self.radioButton_generation_domes.clicked.connect(self.creating_dome)
        self.radioButton_loading_pictures.clicked.connect(self.display_picture)

        # Алгоритм обратки
        # Создание куполов
        self.pushButton_display_domes.clicked.connect(self.creating_dome)
        # Загрузка изображения
        self.pushButton_loading_pictures.clicked.connect(self.load_image)
        # Добавление шума
        self.pushButton_display_noise.clicked.connect(self.noise)
        # Построение спектра
        self.pushButton_building_spectrum.clicked.connect(self.spectrum_numpy)
        # Обновить вид отображения спектра
        self.radioButton_logarithmic_axis.clicked.connect(self.drawing_spectrum_and_zone)
        self.radioButton_linear_axis.clicked.connect(self.drawing_spectrum_and_zone)
        # Восстановление изображения
        self.pushButton_image_recovery.clicked.connect(self.search_zone)

    # ОБРАБОТКА ИНТЕРФЕЙСА
    # Смена режима отображения изображения
    # Выбрано отображение цветного изображения
    def change_picture_to_colored(self, state):
        if state and self.color_picture is not None:
            drawer.image_color_2d(self.graph_1, self.color_picture)

    # Выбрано отображение серого изображения
    def change_picture_to_gray(self, state):
        if state and self.original_picture is not None:
            drawer.image_gray_2d(self.graph_1, self.original_picture)

    # Отобразить изображение
    def display_picture(self):
        # Изображения нет - не отображаем
        if self.color_picture is None:
            return
        self.original_picture = self.gray_picture
        # Проверяем вид отображаемого изображения
        if self.radioButton_color_picture.isChecked():
            drawer.image_color_2d(self.graph_1, self.color_picture)
        else:
            drawer.image_gray_2d(self.graph_1, self.original_picture)

    # Проверка: является ли изображение степени двойки
    def is_image_power_of_two(self, original_picture):
        height, width = original_picture.shape
        if (height & (height - 1)) == 0 and (width & (width - 1)) == 0:
            return original_picture
        else:
            # Проверяем выбор доведения до степени двойки изображения
            window()
            # Проверяем вид отображаемого изображения
            if self.radioButton_interpolation.isChecked():
                original_picture = Processing_Image.interpolate_to_power_of_two(original_picture)
            else:
                original_picture = Processing_Image.resize_to_power_of_two(original_picture)
        return original_picture

    # АЛГОРИТМ РАБОТЫ ПРОГРАММЫ
    # (1 Гауссовы купола) Вычислить график
    def creating_dome(self):
        # Запрашиваем размер области
        width_area = int(self.lineEdit_width_area.text())
        height_area = int(self.lineEdit_height_area.text())
        # Запрашиваем параметры куполов
        # Первый купол
        amplitude_1 = float(self.lineEdit_amplitude_1.text())
        x0_1 = float(self.lineEdit_x0_1.text())
        sigma_x_1 = float(self.lineEdit_sigma_x_1.text())
        y0_1 = float(self.lineEdit_y0_1.text())
        sigma_y_1 = float(self.lineEdit_sigma_y_1.text())
        # Второй купол
        amplitude_2 = float(self.lineEdit_amplitude_2.text())
        x0_2 = float(self.lineEdit_x0_2.text())
        sigma_x_2 = float(self.lineEdit_sigma_x_2.text())
        y0_2 = float(self.lineEdit_y0_2.text())
        sigma_y_2 = float(self.lineEdit_sigma_y_2.text())
        # Третий купол
        amplitude_3 = float(self.lineEdit_amplitude_3.text())
        x0_3 = float(self.lineEdit_x0_3.text())
        sigma_x_3 = float(self.lineEdit_sigma_x_3.text())
        y0_3 = float(self.lineEdit_y0_3.text())
        sigma_y_3 = float(self.lineEdit_sigma_y_3.text())
        # Создаем пустую матрицу пространства
        self.original_picture = np.zeros((height_area, width_area))
        # Для каждой точки матрицы считаем сумму куполов
        for x in range(width_area):
            for y in range(height_area):
                self.original_picture[y, x] = dome_2d(amplitude_1, x, x0_1, sigma_x_1, y, y0_1, sigma_y_1) + \
                                              dome_2d(amplitude_2, x, x0_2, sigma_x_2, y, y0_2, sigma_y_2) + \
                                              dome_2d(amplitude_3, x, x0_3, sigma_x_3, y, y0_3, sigma_y_3)
        # Выводим изображение куполов
        drawer.graph_color_2d(self.graph_1, self.original_picture)

    # (1 Изображение) Загрузить изображение
    def load_image(self):
        # Вызов окна выбора файла
        filename, filetype = QFileDialog.getOpenFileName(None,
                                                         "Выбрать файл изображения",
                                                         ".",
                                                         "All Files(*)")
        # filename = "image_256_256.png"
        # Загружаем изображение
        self.color_picture = cv2.imread(filename, cv2.IMREAD_COLOR)
        # Конвертируем в серый
        self.gray_picture = black_white_image(self.color_picture)
        self.original_picture = self.gray_picture
        # Отображаем изображение
        self.display_picture()

    # (2) Накладываем шум
    def noise(self):
        # Нет исходны данных - сброс
        if self.original_picture is None:
            return
        # Проверка изображения на степень двойки
        self.original_picture = self.is_image_power_of_two(self.original_picture)
        self.noise_image = self.original_picture.copy()
        # Создаем изображение чисто шума
        height, width = self.noise_image.shape
        picture_noise = np.zeros((height, width))
        energy_noise = 0
        for x in range(width):
            for y in range(height):
                val = uniform_distribution()
                # Записываем пиксель шума
                picture_noise[y, x] = val
                # Копим энергию шума
                energy_noise += val * val
        # Запрашиваем процент шума
        noise_percentage = float(self.lineEdit_noise.text()) / 100
        # Считаем энергию зашумленного изображения
        energy_noise_image = energy_pictures(self.noise_image)
        # Считаем коэффициент/множитель шума
        noise_coefficient = math.sqrt(noise_percentage *
                                      (energy_noise_image / energy_noise))
        # К пикселям изображения добавляем пиксель шума
        for x in range(width):
            for y in range(height):
                self.noise_image[y, x] += noise_coefficient * picture_noise[y, x]
        # Отображаем итог
        # Выбираем график или изображение
        if self.radioButton_loading_pictures.isChecked():
            drawer.image_gray_2d(self.graph_2, self.noise_image)
        else:
            drawer.graph_color_2d(self.graph_2, self.noise_image)
        # Считаем разницу эпсилон между исходным и зашумленным изображениями
        epsilon = 0
        height, width = self.original_picture.shape
        for j in range(height):
            for i in range(width):
                epsilon += \
                    (self.original_picture[j, i] - self.noise_image[j, i]) * \
                    (self.original_picture[j, i] - self.noise_image[j, i])
        # Считаем энергию оригинального изображения
        energy_original_picture = energy_pictures(self.original_picture)
        epsilon /= energy_original_picture
        # Вывод эпсилон (разница исходного изображения с зашумленным)
        self.lineEdit_epsilon.setText(f'{epsilon:.6f}')

    # (3) Спектр изображения с шумом. Спектр с диагональной перестановкой
    def spectrum_numpy(self):
        if self.noise_image is None:
            return
        # Перевод изображения в комплексное
        complex_image = np.array(self.noise_image, dtype=complex)
        # Считаем спектр
        self.picture_spectrum = FFT.matrix_fft(complex_image)
        # Берем модуль, для отображения
        module_picture_spectrum = abs(self.picture_spectrum)
        module_picture_spectrum[0, 0] = 0
        # Матрица со спектром посередине
        height, width = module_picture_spectrum.shape
        middle_h = height // 2
        middle_w = width // 2
        self.module_spectrum_repositioned = np.zeros((height, width))
        # Меняем по главной диагонали
        self.module_spectrum_repositioned[0:middle_h, 0:middle_w] = \
            module_picture_spectrum[middle_h:height, middle_w:width]
        self.module_spectrum_repositioned[middle_h:height, middle_w:width] = \
            module_picture_spectrum[0:middle_h, 0:middle_w]
        # Меняем по главной диагонали
        self.module_spectrum_repositioned[middle_h:height, 0:middle_w] = \
            module_picture_spectrum[0:middle_h, middle_w:width]
        self.module_spectrum_repositioned[0:middle_h, middle_w:width] = \
            module_picture_spectrum[middle_h:height, 0:middle_w]
        # Отображаем спектр
        drawer.image_gray_2d(self.graph_3, self.module_spectrum_repositioned,
                             logarithmic_axis=self.radioButton_logarithmic_axis.isChecked())

    # (4.1) Поиск ширины и высоты зоны для заданного процента энергии
    def search_zone(self):
        if self.module_spectrum_repositioned is None:
            return
        height, width = self.module_spectrum_repositioned.shape
        # Энергия модуля спектра
        spectrum_module_energy = energy_pictures(self.module_spectrum_repositioned)
        # Запрашиваем процент энергии
        percent_filtering = float(self.lineEdit_percent_energy.text()) / 100
        # Фильтрация
        # Вычисляем половину высоты и ширины
        half_width = width // 2
        half_height = height // 2
        half_rectangle_width = None
        half_rectangle_height = None
        # Находим большую сторону
        # Если ширина больше или равна высоте
        if width >= height:
            # Начиная с половины ширины до центра перебираем прямоугольники
            for half_rectangle_width in range(half_width, 0, -1):
                # Для данной ширины, находим высоту
                half_rectangle_height = int((half_rectangle_width * half_height) / half_width)
                # Считаем энергию зоны
                zone_energy = 0
                for i in range(half_width - half_rectangle_width, half_width + half_rectangle_width):
                    for j in range(half_height - half_rectangle_height, half_height + half_rectangle_height):
                        zone_energy += self.module_spectrum_repositioned[j, i] * self.module_spectrum_repositioned[j, i]
                # Как только энергия в области стала равна или меньше заданного процента - стоп
                if zone_energy / spectrum_module_energy <= percent_filtering:
                    break
        # Если высота больше ширины
        else:
            # Начиная с половины высоты до центра перебираем прямоугольники
            for half_rectangle_height in range(half_width, 0, -1):
                # Для данной высоты, находим ширину
                half_rectangle_width = int((half_rectangle_height * half_width) / half_height)
                # Считаем энергию зоны
                zone_energy = 0
                for i in range(half_width - half_rectangle_width, half_width + half_rectangle_width):
                    for j in range(half_height - half_rectangle_height, half_height + half_rectangle_height):
                        zone_energy += self.module_spectrum_repositioned[j, i] * self.module_spectrum_repositioned[j, i]
                # Как только энергия в области стала равна или меньше заданного процента - стоп
                if zone_energy / spectrum_module_energy <= percent_filtering:
                    break
        # Находим координаты прямоугольника
        x1 = half_width - half_rectangle_width - 0.5
        y1 = half_height - half_rectangle_height - 0.5
        x2 = half_width + half_rectangle_width - 0.5
        y2 = half_height - half_rectangle_height - 0.5
        x3 = half_width + half_rectangle_width - 0.5
        y3 = half_height + half_rectangle_height - 0.5
        x4 = half_width - half_rectangle_width - 0.5
        y4 = half_height + half_rectangle_height - 0.5
        # Запоминаем размер области заданного процента энергии
        self.half_rectangle_width = half_rectangle_width
        self.half_rectangle_height = half_rectangle_height
        self.energy_boundaries_x = [x1, x2, x3, x4, x1]
        self.energy_boundaries_y = [y1, y2, y3, y4, y1]
        self.drawing_spectrum_and_zone()

    # (4.2) Отрисовка модуля спектра и границ зоны для заданного процента энергии
    def drawing_spectrum_and_zone(self):
        if self.module_spectrum_repositioned is None:
            return
        # Рисуем модуль спектра
        drawer.image_gray_2d(self.graph_3, self.module_spectrum_repositioned,
                             logarithmic_axis=self.radioButton_logarithmic_axis.isChecked())
        # Строим график прямых от точки к точке. Прямоугольник выбранной области
        self.graph_3.axis.plot(self.energy_boundaries_x, self.energy_boundaries_y)
        # Убеждаемся, что все помещается внутри холста
        self.graph_3.figure.tight_layout()
        # Показываем новую фигуру в интерфейсе
        self.graph_3.canvas.draw()
        self.spectrum_nulling()

    # (5) Зануляем комплексный спектр за областью прямоугольника
    def spectrum_nulling(self):
        if (self.half_rectangle_width is None and
                self.half_rectangle_height is None and
                self.picture_spectrum is None):
            return
        height, width = self.picture_spectrum.shape
        for i in range(self.half_rectangle_width, width - self.half_rectangle_width):
            for j in range(0, self.half_rectangle_height):
                self.picture_spectrum[j, i] = 0 + 0j
        for i in range(0, width):
            for j in range(self.half_rectangle_height, height - self.half_rectangle_height):
                self.picture_spectrum[j, i] = 0 + 0j
        for i in range(self.half_rectangle_width, width - self.half_rectangle_width):
            for j in range(height - self.half_rectangle_height, height):
                self.picture_spectrum[j, i] = 0 + 0j
        self.inverse_fourier()

    # (6) Обратное преобразование фурье для восстановления изображения
    def inverse_fourier(self):
        if self.picture_spectrum is None:
            return
        # Считаем спектр
        spectral_reconstructed_image = FFT.matrix_fft_reverse(self.picture_spectrum)
        # Берем модуль, для отображения
        self.module_reconstructed_image = abs(spectral_reconstructed_image)
        drawer.image_gray_2d(self.graph_4, self.module_reconstructed_image)
        self.recovery_parameter()

    # (7) Отображаем разницу между исходным и восстановленным изображением
    def recovery_parameter(self):
        # Проверяем наличие изображений
        if self.module_reconstructed_image is None and self.original_picture is None:
            return
        # Считаем разницу между исходным и восстановленным изображениями
        # Вариант 1
        squared_difference = np.square(self.original_picture - self.module_reconstructed_image)
        recovery_factor = np.sum(squared_difference) / np.sum(np.square(self.original_picture))
        self.lineEdit_recovery_parameter.setText(f'{recovery_factor:.6f}')
        # Вариант 2
        # recovery_factor = 0
        # height, width = self.original_picture.shape
        # for j in range(height):
        #     for i in range(width):
        #         recovery_factor += \
        #             (self.original_picture[j, i] - self.module_reconstructed_image[j, i]) * \
        #             (self.original_picture[j, i] - self.module_reconstructed_image[j, i])
        # # Считаем энергию оригинального изображения
        # energy_original_picture = energy_pictures(self.original_picture)
        # recovery_factor /= energy_original_picture
        # self.lineEdit_recovery_parameter.setText(f'{recovery_factor:.6f}')
