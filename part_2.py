import numpy as np
#Дополнительные задачи

#1.Подсчитать произведение ненулевых элементов на диагонали прямоугольной матрицы.
#Например, для X = np.array([[1, 0, 1], [2, 0, 2], [3, 0, 3], [4, 4, 4]]) ответ 3.

def diagonal_product_numpy(matrix):
    diag = np.diag(matrix)
    non_zero = diag[diag != 0]
    return np.prod(non_zero) if len(non_zero) > 0 else 0

#2.Даны два вектора x и y. Проверить, задают ли они одно и то же мультимножество.
#Например, для x = np.array([1, 2, 2, 4]), y = np.array([4, 2, 1, 2]) ответ True.

def same_multiset_numpy(x, y):
    return np.array_equal(np.sort(x), np.sort(y))

#3.Найти максимальный элемент в векторе x среди элементов, перед которыми стоит ноль.
# Например, для x = np.array([6, 2, 0, 3, 0, 0, 5, 7, 0]) ответ 5.

def max_after_zero_numpy(x):
    zero_indices = np.where(x[:-1] == 0)[0]
    if len(zero_indices) == 0:
        return None
    return np.max(x[zero_indices + 1])

#4.Реализовать кодирование длин серий (Run-length encoding).
# Для некоторого вектора x необходимо вернуть кортеж из двух векторов одинаковой длины.
# Первый содержит числа, а второй - сколько раз их нужно повторить.
# Например, для x = np.array([2, 2, 2, 3, 3, 3, 5]) ответ (np.array([2, 3, 5]), np.array([3, 3, 1])).

def run_length_encoding_numpy(x):
    if len(x) == 0:
        return np.array([]), np.array([])

    positions = np.where(x[1:] != x[:-1])[0] + 1
    starts = np.concatenate(([0], positions))
    ends = np.concatenate((positions, [len(x)]))

    values = x[starts]
    counts = ends - starts

    return values, counts

#5. Даны две выборки объектов - X и Y. Вычислить матрицу евклидовых расстояний между объектами.
#Сравните с функцией scipy.spatial.distance.cdist по скорости работы.
def euclidean_distances_numpy(X, Y):
    X_sq = np.sum(X**2, axis=1, keepdims=True)
    Y_sq = np.sum(Y**2, axis=1, keepdims=True)
    cross_term = -2 * np.dot(X, Y.T)
    return np.sqrt(X_sq + Y_sq.T + cross_term)
