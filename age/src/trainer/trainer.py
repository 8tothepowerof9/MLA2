import os
import torch
import torch.nn as nn
import torch.optim as optim
from src.trainer.base import BaseTrainer
from scripts.utils import EarlyStopping, get_device
import pandas as pd

import time
from src.utils import compute_regression_metrics
from torch.optim.lr_scheduler import StepLR


class AgeTrainer(BaseTrainer):
    def __init__(self, model, config):
        device = get_device(force_cpu=config['train'].get('force_cpu', False))
        super().__init__(model, config, device)
        self.criterion = nn.SmoothL1Loss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=config['train']['lr'])

        self.scheduler = StepLR(
            self.optimizer, 
            step_size=config['train']['lr_step'],
            gamma=config['train']['lr_gamma']
        )

        self.log.update({
            "train_mae": [], "train_r2": [],
             "val_mae": [], "val_r2": []
        })

    def _train_epoch(self, dataloader):
        # Normalize the model [0, 1] for stable regression training
        AGE_MIN = 45.0
        AGE_MAX = 82.0

        self.model.train()
        total_loss = 0.0
        preds, targets = [], []
        total_samples = 0
        batch_loop = 50

        for batch_idx, (images, target, _) in enumerate(dataloader):
            images = images.to(self.device)
            target = target.float().to(self.device)

            # Normalize the target to [0, 1] range
            target_norm = (target - AGE_MIN) / (AGE_MAX - AGE_MIN)

            self.optimizer.zero_grad()
            output = self.model(images).view(-1)
            loss = self.criterion(output, target_norm)
            loss.backward()
            # This ensure that gradients are clipped to avoid exploding gradients
            # prevents unstable updates
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()

            # Denormalize for metric computation
            output_denorm = output.detach() * (AGE_MAX - AGE_MIN) + AGE_MIN
            target_denorm = target

            preds.extend(output_denorm.cpu().numpy().tolist())
            targets.extend(target_denorm.cpu().numpy().tolist())
            total_samples += len(images)

            if (batch_idx + 1) % batch_loop == 0 or (batch_idx + 1) == len(dataloader):
                avg_batch_loss = total_loss / (batch_idx + 1)
                print(f"[Batch {batch_idx+1}/{len(dataloader)}] Samples: {total_samples} | Avg Loss: {avg_batch_loss:.4f}")

        avg_loss = total_loss / len(dataloader)
        mae, r2 = compute_regression_metrics(preds, targets)

        self.log['train_loss'].append(avg_loss)
        self.log['train_mae'].append(mae)
        self.log['train_r2'].append(r2)
        print(f"Train -> MAE: {mae:.4f}, R2: {r2:.4f}")
        return avg_loss

    def _eval_epoch(self, dataloader):
        AGE_MIN = 45.0
        AGE_MAX = 82.0

        self.model.eval()
        total_loss = 0.0
        preds, targets = [], []

        with torch.no_grad():
            for images, target, _ in dataloader:
                images = images.to(self.device)
                target = target.float().to(self.device)

                target_norm = (target - AGE_MIN) / (AGE_MAX - AGE_MIN)

                output = self.model(images).view(-1)
                loss = self.criterion(output, target_norm)
                total_loss += loss.item()

                output_denorm = output * (AGE_MAX - AGE_MIN) + AGE_MIN
                preds.extend(output_denorm.cpu().numpy().tolist())
                targets.extend(target.cpu().numpy().tolist())

        avg_loss = total_loss / len(dataloader)
        mae, r2 = compute_regression_metrics(preds, targets)

        print(f"Predicted range: min={min(preds):.2f}, max={max(preds):.2f}")

        self.log['val_loss'].append(avg_loss)
        self.log['val_mae'].append(mae)
        self.log['val_r2'].append(r2)
        print(f"Validation -> MAE: {mae:.4f}, R2: {r2:.4f}")
        return avg_loss

    def fit(self, train_loader, val_loader):
        print("Training started...")
        epochs = self.config['train']['epochs']
        best_val = float('inf')
        start_time = time.time()

        early_stop = self.config['train'].get('early_stopping', False)
        stopper = EarlyStopping(patience=self.config['train'].get('patience', 5)) if early_stop else None

        model_name = self.config['train']['model_name']
        base_ckpt_dir = self.config['train'].get('checkpoint_dir', 'checkpoints/age')
        save_dir = os.path.join(base_ckpt_dir, model_name)
        os.makedirs(save_dir, exist_ok=True)

        best_model_path = os.path.join(save_dir, "best_model.pt")
        log_path = os.path.join(save_dir, model_name + ".csv")

        for epoch in range(epochs):
            print("----------------------------")
            print(f"Epoch {epoch + 1}/{epochs}")
            epoch_start = time.time()

            train_loss = self._train_epoch(train_loader)
            val_loss = self._eval_epoch(val_loader)

            if hasattr(self, "scheduler"):
                self.scheduler.step()

            elapsed = time.time() - start_time
            epoch_time = time.time() - epoch_start
            remaining = epoch_time * (epochs - epoch - 1)

            improvement = 0.0 if best_val == float('inf') else best_val - val_loss
            best_flag = "[Best]" if val_loss < best_val else ""
            if val_loss < best_val:
                best_val = val_loss
                torch.save(self.model.state_dict(), best_model_path)

            print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f} {best_flag}")
            print(f"Elapsed: {elapsed:.2f}s | Est. Remaining: {remaining:.2f}s | Improvement: {improvement:.4f}")

            if early_stop and stopper(val_loss):
                print("Early stopping triggered. Training halted.")
                break

        pd.DataFrame(self.log).to_csv(log_path, index=False)
        print(f"Training complete. Best model saved to {best_model_path}")
        print(f"Logs saved to {log_path}")