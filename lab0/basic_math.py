import numpy as np
import scipy as sc
from scipy.optimize import minimize_scalar

def matrix_multiplication(matrix_a, matrix_b):
    """
    Задание 1. Функция для перемножения матриц с помощью списков и циклов.
    Вернуть нужно матрицу в формате списка.
    """
    exception = 'Число столбцов первой матрицы должно быть равно числу строк второй матрицы'
    rows_A = len(A)
    cols_B = len(B[0])
    cols_A = len(A[0])

    if cols_A != len(B):
        print(exception)

    C = [[0]*cols_B for i in range(rows_A)]
    for i in range(rows_A):
        for k in range(cols_B):
            for j in range(cols_A):
                C[i][k] += A[i][j] * B[j][k]

    return C 


def F(x, a11, a12, a13):
    return a11 * x**2 + a12 * x + a13

def P(x, a21, a22, a23):
    return a21 * x**2 + a22 * x + a23

def functions(a_1, a_2):
    """
    Задание 2. На вход поступает две строки, содержащие коэффициенты двух функций.
    Необходимо найти точки экстремума функции и определить, есть ли у функций общие решения.
    Вернуть нужно координаты найденных решения списком, если они есть. None, если их бесконечно много.
    """
    a11, a12, a13 = map(float, a_1.split())
    a21, a22, a23 = map(float, a_2.split())

    extremum_F = minimize_scalar(F, args=(a11, a12, a13))
    extremum_P = minimize_scalar(P, args=(a21, a22, a23))
    
    extremum_F_x = extremum_F.x
    extremum_F_value = extremum_F.fun
    extremum_P_x = extremum_P.x
    extremum_P_value = extremum_P.fun
    
    print(f"Экстремум функции F(x) x = {extremum_F_x}, значение F({extremum_F_x}) = {extremum_F_value}")
    print(f"Экстремум функции P(x) x = {extremum_P_x}, значение P({extremum_P_x}) = {extremum_P_value}")
    
    A = a11 - a21
    B = a12 - a22
    C = a13 - a23

    D = B**2 - 4*A*C

    if A == 0 and B == 0 and C == 0:
        return None  
    elif A == 0 and B == 0:
        return []
    elif A == 0:
        x = -C / B
        return [x]
    elif D > 0:
        x1 = (-B + np.sqrt(D)) / (2*A)
        x2 = (-B - np.sqrt(D)) / (2*A)
        return [x1, x2]
    elif D == 0:
        x = -B / (2*A)
        return [x]
    else:
        return []


def calculate_moments(data):
    n = len(data)
    x_mean = np.mean(data)  # выборочное среднее
    m2 = np.sum((data - x_mean) ** 2) / n  # момент 2 порядка (дисперсия)
    m3 = np.sum((data - x_mean) ** 3) / n  # момент 3 порядка
    m4 = np.sum((data - x_mean) ** 4) / n  # момент 4 порядка
    return m2, m3, m4

def skew(x):
    """
    Задание 3. Функция для расчета коэффициента асимметрии.
    Необходимо вернуть значение коэффициента асимметрии, округленное до 2 знаков после запятой.
    """
    m2, m3, _ = calculate_moments(x)
    sigma = np.sqrt(m2)  # стандартное отклонение

    if sigma != 0:
        skew_value = m3 / (sigma ** 3)
        return round(skew_value, 2)  
    else:
        return 0.0  


def kurtosis(x):
    """
    Задание 3. Функция для расчета коэффициента эксцесса.
    Необходимо вернуть значение коэффициента эксцесса, округленное до 2 знаков после запятой.
    """
    m2, _, m4 = calculate_moments(x)
    sigma = np.sqrt(m2)  # стандартное отклонение

    if sigma != 0:
        kurtosis_value = (m4 / (sigma ** 4)) - 3
        return round(kurtosis_value, 2)
    else:
        return 0.0 
