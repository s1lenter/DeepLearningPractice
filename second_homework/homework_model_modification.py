import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from utils import make_regression_data, mse, log_epoch, RegressionDataset

# 1.1 Расширение линейной регрессии
# Модифицируйте существующую линейную регрессию:
# - Добавьте L1 и L2 регуляризацию
# - Добавьте early stopping


class LinearRegression(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.linear = nn.Linear(in_features, 1)

    def forward(self, x):
        return self.linear(x)

# Добавьте L1 и L2 регуляризацию
def regularization(model, l1_c, l2_c):
    l1_reg = 0.0
    l2_reg = 0.0
    for param in model.parameters():
        l1_reg += torch.sum(torch.abs(param))
        l2_reg += torch.sum(param ** 2)
    return l1_c * l1_reg + l2_c * l2_reg

if __name__ == '__main__':
    # Генерируем данные
    X, y = make_regression_data(n=200)

    # Создаём датасет и даталоадер
    dataset = RegressionDataset(X, y)
    val_split = 0.2
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    print(f'Размер train: {len(train_dataset)} | val: {len(val_dataset)}')

    # Создаём модель, функцию потерь и оптимизатор
    model = LinearRegression(in_features=1)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0

    # Обучаем модель
    epochs = 100
    for epoch in range(1, epochs + 1):
        total_train_loss = 0

        for i, (batch_X, batch_y) in enumerate(train_loader):
            optimizer.zero_grad()
            y_pred = model(batch_X)
            loss = criterion(y_pred, batch_y)
            loss += regularization(model, 1e-4, 1e-4) # Вызов функции регуляризации и добавление в результирующий loss
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / (i + 1)

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for j, (val_X, val_y) in enumerate(val_loader):
                val_pred = model(val_X)
                val_loss = criterion(val_pred, val_y)
                total_val_loss += val_loss.item()
        avg_val_loss = total_val_loss / (j + 1)

        log_epoch(epoch, avg_train_loss, val_loss=avg_val_loss)

        # Проверка ранней остановки
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Сохраняем лучшую модель
            torch.save(model.state_dict(), 'linreg_best.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Остановка на эпохе {epoch}")
                break

    # Сохраняем модель
    torch.save(model.state_dict(), 'linreg_torch.pth')

    # Загружаем модель
    new_model = LinearRegression(in_features=1)
    new_model.load_state_dict(torch.load('linreg_torch.pth'))
    new_model.eval()