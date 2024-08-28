import random as rnd
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as sndi
import matplotlib as mpl
from matplotlib.colors import ListedColormap
import time


gist_ncar = mpl.colormaps.get_cmap('gist_ncar').resampled(256)
newcolors = gist_ncar(np.linspace(0, 1, 256))
white = np.array([256 / 256, 256 / 256, 256 / 256, 1])
newcolors[:0, :] = white
newcmp = ListedColormap(newcolors)

n = 100  # размер квадратной решетки.
p = 0.6 # вероятность заполнения узла.
# заполняет случайно квадратную матрицу значениями от 0.00 до 1.00 .
z = np.round(np.random.uniform(0.00, 1.00, (n, n)), 2)
# определяются какие узлы будут заполнены, а какие нет в зависимости от p.
m = z < p
lw, num = sndi.label(m)  # функция sndi.label() маркирует кластеры.


# функция perc_cl убирает все кластеры и оставляет только перколяционный кластер.
def perc_cl(lw):
    # функция intersect1d() возвратит пересечение двух массивов,
    perc_1x = np.intersect1d(lw[0, :], lw[-1, :])
    #  т.е. уникальные элементы, которые встречаются в обоих.
    # функция where() возвратит элементы, удовлетворяющие определённому условию.
    perc = perc_1x[np.where(perc_1x > 0)]

    if len(perc) > 0:
        lw_cluster = lw.copy()

        for i in perc:
            # удаляет не перколяционные кластеры
            lw_cluster[lw_cluster != i] = 0

        fig1 = plt.figure()
        fig1.set_figheight(1)
        fig1.set_figwidth(1)
        ax = fig1.add_subplot(111)
        ax.set_title('Перколяционный кластер', fontsize=3)

        return ax.imshow(lw_cluster, cmap=newcmp, origin='upper')

    else:
        return print('Перколяционный график отсутствует на данной решетке')


# перемешка палитры, чтобы соседние кластеры не сливались с друг другом.
b = np.arange(lw.max() + 1)
rnd.shuffle(b)
shuffled_lw = b[lw]

print(f'Матрица заполненных(True) и незаполненных(False) узлов\n{m}')
print(f'\nМатрица с найденными кластерами\n{lw}')

fig, axes = plt.subplots(1, 2)  # настройка графиков.

perc_cl(lw)

axes[0].imshow(shuffled_lw, cmap=newcmp, origin='upper')
axes[0].set_title('\nНайденые кластеры (каждый кластер отоброжён разным цветом)')

axes[1].imshow(m, cmap=newcmp, origin='upper')
axes[1].set_title('Заполненные и незаполненные узлы\n(белые клетки - заполненные узлы)')

fig.set_figwidth(12)  # ширина и
fig.set_figheight(6)  # высота "Figure"

plt.show()


def P(size):
    p = np.linspace(0.4, 1.0, 100)  # список вероятностей заполнения узла
    nx = len(p)
    Ni = np.zeros(nx)
    N = 1000  # общее кол-во симуляций
    L = size  # размер квадратной решётки

    for i in range(N):
        # заполняет случайно квадратную матрицу значениями от 0.00 до 1.00 .
        z = np.round(np.random.uniform(0.00, 1.00, (L, L)), 2)

        for ip in range(nx):
            # определяются какие узлы будут заполнены, а какие нет в зависимости от p.
            m = z < p[ip]
            lw, num = sndi.label(m)  # функция sndi.label() маркирует кластеры.
            # функция intersect1d() возвратит пересечение двух массивов,
            perc_x = np.intersect1d(lw[0, :], lw[-1, :])  # т.е. уникальные элементы, которые встречаются в обоих.

            # функция where() возвратит элементы, удовлетворяющие определённому условию.
            perc = perc_x[np.where(perc_x > 0)]

            if len(perc) > 0:
                Ni[ip] = Ni[ip] + 1  # подсчёт кол-ва возникновения перколяции

    Pi = Ni / N  # нахождение вероятности появление перколяции в зависимости от размера решетки

    plt.plot(p, Pi, label=f'L={L}')
    plt.xlabel('$p$')
    plt.ylabel('$Pi$')
    plt.legend()


P(50)
P(100)
P(200)

plt.show()


def threshold():
    h = np.array([0.001], float)  # шаг уменьшения концентрации
    N = 100  # общее кол-во симуляций
    L = 1000  # размер решетки
    all_pc = np.array([], float)  # массив с наименьшими концентрациями

    for i in range(N):

        start_p = np.array([0.6500], float)  # стартовая концентрация
        perc = np.array([0.0000], float)
        print(f'{i} %')

        while len(perc) > 0:

            z = np.round(np.random.uniform(0.00, 1.00, (L, L)), 2)
            m = z < start_p
            lw, num = sndi.label(m)  # функция sndi.label() маркирует кластеры.
            # функция intersect1d() возвратит пересечение двух массивов,
            perc_x = np.intersect1d(lw[0, :], lw[-1, :])  # т.е. уникальные элементы, которые встречаются в обоих.
            # функция where() возвратит элементы, удовлетворяющие определённому условию.
            perc = perc_x[np.where(perc_x > 0)]

            start_p = np.subtract(start_p, h)  # если в квадратной решетке есть перкол. кластер,
            # то уменьшаем концентрацию заполненных узлов на 0.0001

        else:
            start_p = start_p + h  # если в кв. решетке нету перкол. кластера, то увеличиваем концентрацию
            # заполненных узлов на 0.0001 и заносим значение в таблицу
            all_pc = np.append(all_pc, start_p)

            # усредняем все полученные значения и выводим среднее значение порога перколяции
    return print(f'Порог перколяции для квадратной решетки равен {np.round(np.mean(all_pc), 5)} с размером {L}')


start_time = time.time()
threshold()
print("--- %s seconds ---" % (time.time() - start_time))
