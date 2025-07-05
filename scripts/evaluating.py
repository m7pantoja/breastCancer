import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    classification_report
)

def get_metrics(model, val_loader, device, task, class_names=None):

    if task == 'binary':

        model.eval()
        all_logits = []
        all_labels = []

        with torch.no_grad(): # no calculamos gradientes en el proceso de evaluación
            for images, labels in val_loader:
                images = images.to(device)
                logits = model(images).cpu().numpy().ravel()
                all_logits.append(logits)
                all_labels.append(labels.numpy().ravel())

        all_logits = np.concatenate(all_logits)
        all_labels = np.concatenate(all_labels)

        all_probs = 1 / (1 + np.exp(-all_logits)) # pasamos los logits por la función sigmoide para obtener las probabilidades de clase
        all_preds = (all_probs > 0.5).astype(int) # fijamos el threshold en 0.5 para determinar la predicción de clase

        metrics = { 
            "Accuracy":  accuracy_score(all_labels, all_preds),
            "Precision": precision_score(all_labels, all_preds),
            "Recall":    recall_score(all_labels, all_preds),
            "F1":        f1_score(all_labels, all_preds),
            "ROC_AUC":   roc_auc_score(all_labels, all_probs)
        } # obtenemos las principales métricas

        cm = confusion_matrix(all_labels, all_preds) 
        fpr, tpr, _ = roc_curve(all_labels, all_probs) # coordenadas para construir la Curva ROC

        return metrics, cm, (fpr, tpr)
    
    elif task == 'multiclass':
        
        model.eval()
        all_preds = []
        all_labels = []
    
        with torch.no_grad(): # no calculamos gradientes en el proceso de evaluación
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)  
                preds = torch.argmax(outputs, dim=1) # obtención de las predicciones de clase por batch

                all_preds.append(preds.cpu())
                all_labels.append(labels.cpu())

        all_preds = torch.cat(all_preds).numpy()
        all_labels = torch.cat(all_labels).numpy()

        creport = classification_report(all_labels,all_preds,target_names=class_names) # incluye métricas separadas, ponderadas y macro
        cm = confusion_matrix(all_labels,all_preds)

        return creport, cm

    else:
        raise ValueError("Argumento inválido para el parámetro task. Debe ser 'binary' o 'multiclass'")