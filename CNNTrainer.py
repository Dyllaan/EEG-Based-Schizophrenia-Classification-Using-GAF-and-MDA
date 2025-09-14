import torch
import torch.nn as nn
import numpy as np
import copy

class CNNTrainer:
    def __init__(self, lr=1e-3, patience=5, min_lr=1e-5, early_stopping_patience=7, 
                 scheduler_type='plateau', optimizer_type='adam', warmup_epochs=5):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.lr = lr
        self.patience = patience
        self.min_lr = min_lr
        self.early_stopping_patience = early_stopping_patience
        self.scheduler_type = scheduler_type
        self.optimizer_type = optimizer_type
        self.warmup_epochs = warmup_epochs
        self.model = None
        self.criterion = None
        self.optimiser = None
        self.scheduler = None
        self.warmup_scheduler = None
        self.scaler = None


    def setup(self, model, epochs=None):
        print(f"Using device: {self.device}")
        print(f"Using optimizer: {self.optimizer_type}, scheduler: {self.scheduler_type}")
        self.model = model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        if self.optimizer_type == 'adam':
            self.optimiser = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=1e-4)
        elif self.optimizer_type == 'sgd':
            self.optimiser = torch.optim.SGD(model.parameters(), lr=self.lr, momentum=0.9, weight_decay=1e-4)
        if self.scheduler_type == 'plateau':
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimiser, mode='min', factor=0.1, patience=self.patience, min_lr=self.min_lr, threshold=1e-3
            )
        elif self.scheduler_type == 'cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimiser, T_max=epochs - self.warmup_epochs, eta_min=self.min_lr
            )
        if self.warmup_epochs > 0:
            self.warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                self.optimiser, start_factor=0.01, end_factor=1.0, total_iters=self.warmup_epochs
            )
        self.scaler = torch.amp.GradScaler() if self.device.type == 'cuda' else None

    def run_epoch(self, loader, training=True):
        if training:
            self.model.train()
        else:
            self.model.eval()
        total_loss, correct, total = 0, 0, 0

        for batch_data, batch_labels in loader:
            batch_data = batch_data.to(self.device)
            batch_labels = batch_labels.to(self.device)

            if training:
                self.optimiser.zero_grad()
                if self.scaler and self.device.type == 'cuda':
                    with torch.amp.autocast(self.device.type):
                        outputs = self.model(batch_data)
                        loss = self.criterion(outputs, batch_labels)
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimiser)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.scaler.step(self.optimiser)
                    self.scaler.update()
                else:
                    outputs = self.model(batch_data)
                    loss = self.criterion(outputs, batch_labels)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimiser.step()
            else:
                with torch.no_grad():
                    if self.scaler and self.device.type == 'cuda':
                        with torch.amp.autocast(self.device.type):
                            outputs = self.model(batch_data)
                            loss = self.criterion(outputs, batch_labels)
                    else:
                        outputs = self.model(batch_data)
                        loss = self.criterion(outputs, batch_labels)

            total_loss += loss.item()
            batch_correct, batch_total = self.calc_acc(outputs, batch_labels)
            correct += batch_correct
            total += batch_total
        avg_loss = total_loss / len(loader)
        acc = correct / total
        return avg_loss, acc

    def train(self, model, train_loader, val_loader, epochs):
        self.setup(model, epochs)
        best_val_loss = float('inf')
        best_model_state = None
        best_epoch = 0
        epochs_without_improvement = 0

        print(f"Training for {epochs} epochs (early stopping: {self.early_stopping_patience})")
        print(f"Results in format of loss/acc for both train and validation sets")
        print(f"Initial LR: {self.optimiser.param_groups[0]['lr']:.6f}")

        for epoch in range(epochs):
            train_loss, train_acc = self.run_epoch(train_loader, training=True)
            val_loss, val_acc = self.run_epoch(val_loader, training=False)
            # Update learning rate
            prev_lr = self.optimiser.param_groups[0]['lr']
            if self.warmup_scheduler and epoch < self.warmup_epochs:
                self.warmup_scheduler.step()
            elif self.scheduler_type == 'cosine':
                self.scheduler.step()
            else:
                self.scheduler.step(val_loss)
            current_lr = self.optimiser.param_groups[0]['lr']
            # Improved loss?
            if val_loss < best_val_loss - 1e-3:
                best_val_loss = val_loss
                best_epoch = epoch + 1
                best_model_state = copy.deepcopy(model.state_dict())
                epochs_without_improvement = 0
                improved = True
            else:
                epochs_without_improvement += 1
                improved = False

            if (epoch + 1) % 5 == 0 or improved:
                self.log_epoch(epoch, epochs, train_loss, train_acc, val_loss, val_acc, current_lr, improved)
            if prev_lr != current_lr:
                print(f"LR reduced: {prev_lr:.6f} → {current_lr:.6f}")

            if epochs_without_improvement >= self.early_stopping_patience:
                print(f"Early stopping after {self.early_stopping_patience} epochs without improvement")
                break

        print(f"Training done, best validation loss: {best_val_loss:.4f} at epoch {best_epoch}")
        model.load_state_dict(best_model_state)
        # Final validation
        final_loss, final_acc = self.run_epoch(val_loader, training=False)
        print(f"Restored model validation accuracy: {final_acc:.4f}, validation loss: {final_loss:.4f}")
        return model

    def evaluate(self, test_loader):
        self.model.eval()
        correct = 0
        total = 0
        all_predictions = []
        all_labels = []
        with torch.no_grad():
            for batch_data, batch_labels in test_loader:
                batch_data = batch_data.to(self.device)
                batch_labels = batch_labels.to(self.device)
                outputs = self.model(batch_data)
                batch_correct, batch_total = self.calc_acc(outputs, batch_labels)
                correct += batch_correct
                total += batch_total
                predicted = outputs.max(1)[1]
                all_predictions.append(predicted)
                all_labels.append(batch_labels)
        predictions = torch.cat(all_predictions).cpu().numpy()
        true_labels = torch.cat(all_labels).cpu().numpy()
        return correct / total, predictions, true_labels

    def log_epoch(self, epoch, epochs, train_loss, train_acc, val_loss, val_acc, current_lr, improved):
        status = " - lowest loss so far" if improved else ""
        print(f'Epoch [{epoch+1:2d}/{epochs}] | '
              f'Train: {train_loss:.4f}/{train_acc:.4f} | '
              f'Val: {val_loss:.4f}/{val_acc:.4f} | '
              f'LR: {current_lr:.6f}{status}')

    def calc_acc(self, outputs, labels):
        predicted = outputs.max(1)[1]
        return predicted.eq(labels).sum().item(), labels.size(0)