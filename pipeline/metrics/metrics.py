from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import torch

# Функция для вычисления метрик
def compute_metrics(y_true, y_pred, threshold=0.5):
    y_true = y_true.cpu().detach().float().numpy()  # Переводим на CPU
    y_pred = y_pred.cpu().detach().float().numpy()  # Переводим на CPU

    # Бинаризуем предсказания (0 или 1)
    y_pred_binary = (y_pred > threshold).astype(int)

    # print(y_pred.shape, y_true.shape)
    # Вычисляем метрики
    accuracy = accuracy_score(y_true, y_pred_binary)
    precision = precision_score(y_true, y_pred_binary)
    recall = recall_score(y_true, y_pred_binary)
    f1 = f1_score(y_true, y_pred_binary)
    roc_auc = roc_auc_score(y_true, y_pred)  # AUC по вероятностям

    return accuracy, precision, recall, f1, roc_auc