import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from second_homework.logreg_modification import LogisticRegression

# 2.1 Кастомный Dataset класс
# Создайте кастомный класс датасета для работы с CSV файлами:
# - Загрузка данных из файла
# - Предобработка (нормализация, кодирование категорий)
# - Поддержка различных форматов данных (категориальные, числовые, бинарные и т.д.)
class CSVDataset(Dataset):
    def __init__(self, csv_path, target_name, cats=None, binaries=None):
        self.data = pd.read_csv(csv_path)
        self.target_col = target_name
        self.cat_cols = cats if cats else []
        self.binary_cols = binaries if binaries else []

        # Выделяем целевую переменную
        self.y = torch.tensor(self.data[target_name].values, dtype=torch.float32).unsqueeze(1)

        # Убираем целевую колонку из данных
        X_data = self.data.drop(columns=[target_name])

        # Работа с категориями
        if self.cat_cols:
            cat_data = X_data[self.cat_cols]
            encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
            cat_encoded = encoder.fit_transform(cat_data)
            self.cat_tensor = torch.tensor(cat_encoded, dtype=torch.float32)
        else:
            self.cat_tensor = None

        # Работа с бинарными колонками
        if self.binary_cols:
            bin_data = X_data[self.binary_cols].astype(float)
            self.bin_tensor = torch.tensor(bin_data.values, dtype=torch.float32)
        else:
            self.bin_tensor = None

        # Работа с числовыми колонками
        num_cols = [col for col in X_data.columns if col not in self.cat_cols + self.binary_cols]
        if num_cols:
            num_data = X_data[num_cols]
            scaler = StandardScaler()
            num_scaled = scaler.fit_transform(num_data)
            self.num_tensor = torch.tensor(num_scaled, dtype=torch.float32)
        else:
            self.num_tensor = None

        tensors = [t for t in [self.cat_tensor, self.bin_tensor, self.num_tensor] if t is not None]
        self.X = torch.cat(tensors, dim=1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# Найдите csv датасеты для регрессии и бинарной классификации и, применяя наработки из предыдущей части задания, обучите линейную и логистическую регрессию

# Выбрал датасеты deabets для линейной регрессии и breast_cancer для бинарной классификации
from sklearn.datasets import load_diabetes, load_breast_cancer


# Данные Diabetes для регрессии
diabetes = load_diabetes()
X_diabetes = pd.DataFrame(diabetes['data'], columns=diabetes['feature_names'])
y_diabetes = pd.Series(diabetes['target'], name="target")
df_diabetes = pd.concat([X_diabetes, y_diabetes], axis=1)
df_diabetes.to_csv("diabetes.csv", index=False)

# Подготовка данных и обучение линейной регерессии
reg_dataset = CSVDataset("diabetes.csv", target_col="target")
reg_loader = DataLoader(reg_dataset, batch_size=32, shuffle=True)

model = nn.Linear(reg_dataset.X.shape[1], 1)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

epochs = 100
for epoch in range(1, epochs+1):
    total_loss = 0
    for X_batch, y_batch in reg_loader:
        optimizer.zero_grad()
        y_pred = model(X_batch)
        loss = criterion(y_pred, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(reg_loader)
    if epoch % 10 == 0:
        print(f"[Diabetes] Epoch {epoch}: loss={avg_loss:.4f}")

torch.save(model.state_dict(), "diabetes_model.pth")


# Данные Breast Cancer для бинарной классификации
cancer = load_breast_cancer()
X_cancer = pd.DataFrame(cancer['data'], columns=cancer['feature_names'])
y_cancer = pd.Series(cancer['target'], name="target")
df_cancer = pd.concat([X_cancer, y_cancer], axis=1)
df_cancer.to_csv("breast_cancer.csv", index=False)

# Подготовка данных и обучение бинарной классификации
clf_dataset = CSVDataset("breast_cancer.csv", target_col="target")
clf_loader = DataLoader(clf_dataset, batch_size=32, shuffle=True)

model = LogisticRegression(clf_dataset.X.shape[1],1)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

epochs = 100
for epoch in range(1, epochs+1):
    total_loss, total_acc = 0, 0
    for X_batch, y_batch in clf_loader:
        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()
        y_pred = torch.sigmoid(logits) > 0.5
        acc = (y_pred == y_batch).float().mean().item()
        total_loss += loss.item()
        total_acc += acc
    avg_loss = total_loss / len(clf_loader)
    avg_acc = total_acc / len(clf_loader)
    if epoch % 10 == 0:
        print(f"[BreastCancer] Epoch {epoch}: loss={avg_loss:.4f}, acc={avg_acc:.4f}")

torch.save(model.state_dict(), "breast_cancer_model.pth")