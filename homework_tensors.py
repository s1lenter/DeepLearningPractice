import torch

# 1.1 Создание тензоров
# Создайте следующие тензоры:
# - Тензор размером 3x4, заполненный случайными числами от 0 до 1
tensor1 = torch.rand(3, 4)
print(tensor1)

# - Тензор размером 2x3x4, заполненный нулями
tensor2 = torch.zeros(2,3,4)
print(tensor2)

# - Тензор размером 5x5, заполненный единицами
tensor3 = torch.ones(5,5)
print(tensor3)

# - Тензор размером 4x4 с числами от 0 до 15 (используйте reshape)
tensor4 = torch.reshape(torch.arange(16), (4,4))
print(tensor4)



# 1.2 Операции с тензорами
# Дано: тензор A размером 3x4 и тензор B размером 4x3

A = torch.rand(3, 4)
B = torch.rand(4, 3)

print(A)
print(B)

# Выполните:
# - Транспонирование тензора A
a_transpose = A.T

# - Матричное умножение A и B
matrix_multiply = A @ B
print(matrix_multiply)

# - Поэлементное умножение A и транспонированного B
multiply = A * B.T
print(multiply)

# - Вычислите сумму всех элементов тензора A
print(A.sum())



# 1.3 Индексация и срезы
# Создайте тензор размером 5x5x5
tensor5 = torch.rand(5, 5, 5)
print(tensor5)
# Извлеките:
# - Первую строку
print(tensor5[0,0,:])
# - Последний столбец
print(tensor5[:, :, -1])
# - Подматрицу размером 2x2 из центра тензора
print(tensor5[3:4, 2:4, 2:4])
# - Все элементы с четными индексами
print(tensor5[::2, ::2, ::2])




# # 1.4 Работа с формами
# # Создайте тензор размером 24 элемента
# tensor6 = torch.rand(24)
# print(tensor6)
#
# # Преобразуйте его в формы:
# # - 2x12
# tensor6_2x12 = tensor6.view(2, 12)
# print(tensor6_2x12)
#
# # - 3x8
# tensor6_3x8 = tensor6.view(3, 8)
# print(tensor6_3x8)
#
# # - 4x6
# tensor6_4x6 = tensor6.view(4, 6)
# print(tensor6_4x6)
#
# # - 2x3x4
# tensor6_2x3x4 = tensor6.view(2, 3, 4)
# print(tensor6_2x3x4)
#
# # - 2x2x2x3
# tensor6_2x2x2x3 = tensor6.view(2, 2, 2, 3)
# print(tensor6_2x2x2x3)