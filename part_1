import numpy as np

#Упражнения по библиотеке Numpy

#1. Дан случайный массив, поменять знак у элементов, значения которых между 3 и 8
def change_sign_numpy(arr):
    mask = (arr > 3) & (arr < 8)
    arr[mask] = -arr[mask]
    return arr

#2. Заменить максимальный элемент случайного массива на 0
def replace_max_numpy(arr):
    arr[arr == np.max(arr)] = 0
    return arr

#3. Построить прямое произведение массивов (все комбинации с каждым элементом). На вход подается двумерный массив
def cartesian_product_numpy(arrays):
    return np.array(np.meshgrid(*arrays)).T.reshape(-1, len(arrays))

#4. Даны 2 массива A (8x3) и B (2x2). Найти строки в A, которые содержат элементы из каждой строки в B, независимо от порядка элементов в B
def find_rows_numpy(A, B):
    mask = np.all([np.any(np.isin(A, b_row).reshape(A.shape[0], -1), axis=1) 
                   for b_row in B], axis=0)
    return np.where(mask)[0]

#5. Дана 10x3 матрица, найти строки из неравных значений (например строка [2,2,3] остается, строка [3,3,3] удаляется)
def unequal_rows_numpy(matrix):
    mask = ~np.all(matrix == matrix[:, [0]], axis=1)
    return matrix[mask]

#6. Дан двумерный массив. Удалить те строки, которые повторяются
def remove_duplicates_numpy(arr):
    return np.unique(arr, axis=0)
