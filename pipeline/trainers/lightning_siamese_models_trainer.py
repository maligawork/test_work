import torch
import numpy as np
from .lightning_trainer import LightningTrainerSiamese

from pipeline.metrics.metrics import compute_metrics


class SiameseLightningModel(LightningTrainerSiamese):
    """Siamese models training/validation lightning module."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, img1, img2):
        return self.model(img1, img2)

    def on_train_start(self):
        super().on_train_start()
        self.loss_fn.to(self.device)

    def on_validation_start(self):
        self.loss_fn.to(self.device)

    def on_test_start(self):
        self.loss_fn.to(self.device)

    def on_train_epoch_end(self):
        self.shared_epoch_end(self.training_step_outputs, "train")
        self.training_step_outputs.clear()
        return
    
    def on_validation_epoch_end(self):
        self.shared_epoch_end(self.validation_step_outputs, "valid")
        self.validation_step_outputs.clear()
        return
    
    def on_test_epoch_end(self):
        self.shared_epoch_end(self.test_step_outputs, "test")
        self.test_step_outputs.clear()
        return

    def training_step(self, batch, batch_idx):
        images, labels = batch
        logits = self(images[0], images[1])

        loss = self.loss_fn(logits, labels)

        outputs = torch.sigmoid(logits)
        # Вычисляем метрики
        accuracy, precision, recall, f1, roc_auc = compute_metrics(
            labels, 
            outputs,
            threshold=self.metrics_threshold,
        )

        return {
            'loss': loss / self.devices_count,
            'accuracy': accuracy,
            'precision': precision, 
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc,
            'pred': outputs,
        }

    def validation_step(self, batch, batch_idx):
        images, labels = batch

        logits = self(images[0], images[1])

        val_loss = self.loss_fn(logits, labels)

        outputs = torch.sigmoid(logits)
        # Вычисляем метрики
        accuracy, precision, recall, f1, roc_auc = compute_metrics(
            labels, 
            outputs,
            threshold=self.metrics_threshold,
        )

        return {
            'loss': val_loss / self.devices_count,
            'accuracy': accuracy,
            'precision': precision, 
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc,
            'pred': outputs,
        }

    def test_step(self, batch, batch_idx):
        images, labels = batch

        logits = self(images[0], images[1])

        test_loss = self.loss_fn(logits, labels)

        outputs = torch.sigmoid(logits)
        # Вычисляем метрики
        accuracy, precision, recall, f1, roc_auc = compute_metrics(
            labels, 
            outputs,
            threshold=self.metrics_threshold,
        )

        return {
            'loss': test_loss / self.devices_count,
            'accuracy': accuracy,
            'precision': precision, 
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc,
            'pred': outputs,
        }
    

    def shared_epoch_end(self, outputs, stage):
        # aggregate step metics
        loss = torch.mean(torch.tensor([x["loss"] for x in outputs]))
        accuracy = torch.mean(torch.tensor([x["accuracy"] for x in outputs]))
        precision = torch.mean(torch.tensor([x["precision"] for x in outputs]))
        recall = torch.mean(torch.tensor([x["recall"] for x in outputs]))
        f1 = torch.mean(torch.tensor([x["f1"] for x in outputs]))
        roc_auc = torch.mean(torch.tensor([x["roc_auc"] for x in outputs]))
        
    
        metrics = {
            f'{stage}_loss': loss,
            f'{stage}_accuracy': accuracy,
            f'{stage}_precision': precision, 
            f'{stage}_recall': recall,
            f'{stage}_f1': f1,
            f'{stage}_roc_auc': roc_auc,
        }

        self.log_dict(metrics, prog_bar=True, sync_dist=True)