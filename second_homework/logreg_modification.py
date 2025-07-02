import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, random_split
from utils import make_classification_data, ClassificationDataset, log_epoch, accuracy
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 1.2 Расширение логистической регрессии
# Модифицируйте существующую логистическую регрессию:
# - Добавьте поддержку многоклассовой классификации
# - Реализуйте метрики: precision, recall, F1-score, ROC-AUC
# - Добавьте визуализацию confusion matrix

class LogisticRegression(nn.Module):
    def __init__(self, in_features, num_classes):
        super().__init__()
        self.linear = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.linear(x)


def compute_metrics(y_true, y_pred_logits):
    y_pred_labels = y_pred_logits.argmax(dim=1).cpu().numpy()
    y_true = y_true.cpu().numpy()

    precision = precision_score(y_true, y_pred_labels)
    recall = recall_score(y_true, y_pred_labels)
    f1 = f1_score(y_true, y_pred_labels)

    try:
        y_true_oh = np.eye(np.max(y_true)+1)[y_true]
        roc_auc = roc_auc_score(y_true_oh, y_pred_logits.softmax(dim=1).cpu().numpy(), multi_class='ovr')
    except ValueError:
        roc_auc = np.nan  # если данные некорректны для ROC-AUC

    return precision, recall, f1, roc_auc

def plot_confusion_matrix(y_true, y_pred_labels, labels):
    cm = confusion_matrix(y_true, y_pred_labels)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()

# Генерируем данные
X, y = make_classification_data(n=200)

# Разбиваем на train/val
X_train, X_val, y_train, y_val = train_test_split(X.numpy(), y.numpy(), test_size=0.2)

train_dataset = ClassificationDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
val_dataset = ClassificationDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

print(f'Размер train: {len(train_dataset)} | val: {len(val_dataset)}')

# Создаём модель, функцию потерь и оптимизатор
model = LogisticRegression(2, 1)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# Обучаем модель
epochs = 100
for epoch in range(1, epochs + 1):
    model.train()
    total_loss = 0
    total_acc = 0

    for i, (batch_X, batch_y) in enumerate(train_loader):
        optimizer.zero_grad()
        logits = model(batch_X)
        loss = criterion(logits, batch_y)
        loss.backward()
        optimizer.step()

        y_pred = torch.sigmoid(logits)
        acc = accuracy(y_pred, batch_y)

        total_loss += loss.item()
        total_acc += acc

    avg_loss = total_loss / len(train_loader)
    avg_acc = total_acc / len(train_loader)

    model.eval()
    all_preds, all_targets = [], []

    with torch.no_grad():
        for val_X, val_y in val_loader:
            logits = model(val_X)
            all_preds.append(torch.sigmoid(logits))
            all_targets.append(val_y)

    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets).long().squeeze()

    precision, recall, f1, roc_auc = compute_metrics(all_targets, all_preds)
    log_epoch(epoch, avg_loss, acc=avg_acc, precision=precision, recall=recall, f1=f1, roc_auc=roc_auc)

    if epoch == epochs:
        y_pred_labels = (all_preds > 0.5).int().cpu().numpy()
        plot_confusion_matrix(all_targets.cpu().numpy(), y_pred_labels, labels=["0", "1"])

# Сохраняем модель
torch.save(model.state_dict(), 'logreg_torch.pth')

# Загружаем модель
new_model = LogisticRegression(2, 1)
new_model.load_state_dict(torch.load('logreg_torch.pth'))
new_model.eval()