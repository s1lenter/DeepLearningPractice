import math
import torch

# 2.1 Простые вычисления с градиентами
# Создайте тензоры x, y, z с requires_grad=True
x = torch.tensor(1.0, requires_grad=True)
y = torch.tensor(2.0, requires_grad=True)
z = torch.tensor(3.0, requires_grad=True)

# Вычислите функцию: f(x,y,z) = x^2 + y^2 + z^2 + 2*x*y*z
f = x**2 + y**2 + z**2 + 2*x*y*z
f.backward()
print(f)
# Найдите градиенты по всем переменным
print(x.grad, y.grad, z.grad)
# Проверьте результат аналитически
# df/dx = 2x + 2yz = 14
# df/dy = 2y + 2xz = 10
# df/dy = 2z + 2xy = 10



# 2.2 Градиент функции потерь
# Реализуйте функцию MSE (Mean Squared Error):
# MSE = (1/n) * Σ(y_pred - y_true)^2
def mse(y_pred, y_true):
    return torch.mean((y_pred - y_true) ** 2)

# где y_pred = w * x + b (линейная функция)
y_true = torch.rand(5)
print(y_true)

x = torch.rand(5, requires_grad=True)

w = torch.tensor(0.0, requires_grad=True)
b = torch.tensor(0.0, requires_grad=True)

y_pred = w * x + b
loss = mse(y_pred, y_true)
loss.backward()
# Найдите градиенты по w и b
print(w.grad, b.grad)


# 2.3 Цепное правило
# Реализуйте составную функцию: f(x) = sin(x^2 + 1)
x = torch.tensor(45.0, requires_grad=True)
f = torch.sin(x**2 - 1)
f.backward()
# Найдите градиент df/dx
df_dx = x.grad
print(df_dx)
hand_df_dx = torch.cos(x**2 - 1) * 2*x
print(hand_df_dx)
# Проверьте результат с помощью torch.autograd.grad
autograd_df_dx = torch.autograd.grad(torch.sin(x**2 - 1), x)
print(autograd_df_dx)
