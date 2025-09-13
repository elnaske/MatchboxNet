import torch
from torch import nn
from torch.optim.lr_scheduler import LambdaLR
from torch.nn.functional import softmax
from torch.amp import autocast, GradScaler
from tqdm.notebook import tqdm
from collections import defaultdict
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import numpy as np

from torch_optimizer import NovoGrad

class Trainer():
    def __init__(self, model, class_weights = None, device = None, random_state=None):
        if random_state is not None:
            torch.manual_seed(random_state)
            torch.cuda.manual_seed(random_state)

        if device is not None:
            self.device = device
        else:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.model = model.to(self.device)

        self.optimizer = None
        self.scheduler = None

        self.loss_fn = nn.CrossEntropyLoss(weight=class_weights.to(self.device))

        self.prog = defaultdict(list)

        self.metrics = ['loss', 'acc', 'f1', 'prec', 'rec']

    def fit(self, train_dataloader, eval_dataloader, epochs, max_lr, min_lr, verbose=False, print_freq=10):
        if self.optimizer is None:
            self.optimizer = NovoGrad(params=self.model.parameters(), lr=max_lr, betas = (0.95, 0.5), weight_decay=0.001)

        if self.scheduler is None:
            total_steps = epochs * len(train_dataloader)
            lr_schedule = get_lr_schedule(total_steps, warmup_ratio=0.05, hold_ratio=0.45, max_lr=max_lr, min_lr=min_lr)
            self.scheduler = LambdaLR(self.optimizer, lr_schedule)

        self.scaler = GradScaler(self.device)

        # Training Loop
        for epoch in tqdm(range(epochs)):
            train_metrics = self.step(train_dataloader, train=True)
            eval_metrics = self.step(eval_dataloader, train=False)

            for value, metric in zip(train_metrics, self.metrics):
                self.log(f'train_{metric}', value)
            for value, metric in zip(eval_metrics, self.metrics):
                self.log(f'eval_{metric}', value)

            if verbose and (epoch + 1) % print_freq == 0:
                train_loss, train_acc, train_f1, train_prec, train_rec = train_metrics
                eval_loss, eval_acc, eval_f1, eval_prec, eval_rec = eval_metrics

                print(f'Epoch {epoch + 1}')
                print(f'Train: Loss = {train_loss:.2f} | Acc. = {train_acc:.2f} | F1 = {train_f1:.2f} | Prec. = {train_prec:.2f} | Rec. = {train_rec:.2f}')
                print(f'Eval: Loss = {eval_loss:.2f} | Acc. = {eval_acc:.2f} | F1 = {eval_f1:.2f} | Prec. = {eval_prec:.2f} | Rec. = {eval_rec:.2f}')

    def step(self, dataloader, train=False):
        avg_loss = 0
        avg_acc = 0
        avg_f1 = 0
        avg_prec = 0
        avg_rec = 0

        for X, y in tqdm(dataloader):
            X, y = X.to(self.device), y.to(self.device)

            if train:
                self.optimizer.zero_grad()

                with autocast(self.device):
                    y_pred = self.model(X)
                    loss = self.loss_fn(y_pred, y)

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scheduler.step()
                self.scaler.update()

            else:
                with torch.no_grad():
                    y_pred = self.model(X)
                    loss = self.loss_fn(y_pred, y)

            avg_loss += loss.item()

            y_pred_sm = softmax(y_pred, dim = -1).argmax(-1)
            y_pred_np = y_pred_sm.cpu().numpy()
            y = y.cpu().numpy()

            acc = accuracy_score(y, y_pred_np)
            f1 = f1_score(y, y_pred_np, average='macro', zero_division=np.nan)
            prec = precision_score(y, y_pred_np, average='macro', zero_division=np.nan)
            rec = recall_score(y, y_pred_np, average='macro', zero_division=np.nan)

            avg_acc += acc
            avg_f1 += f1
            avg_prec += prec
            avg_rec += rec

        avg_loss /= len(dataloader)
        avg_acc /= len(dataloader)
        avg_f1 /= len(dataloader)
        avg_prec /= len(dataloader)
        avg_rec /= len(dataloader)

        return avg_loss, avg_acc, avg_f1, avg_prec, avg_rec

    def log(self, metric, value):
        self.prog[metric].append(value.item() if isinstance(value, torch.Tensor) else value)

    def evaluate(self, test_dataloader):
        test_metrics = self.step(test_dataloader, train=False)

        for value, metric in zip(test_metrics, self.metrics):
            self.log(f'test_{metric}', value)

        test_loss, test_acc, test_f1, test_prec, test_rec = test_metrics

        print(f'Test: Loss = {test_loss:.2f} | Acc. = {test_acc:.2f} | F1 = {test_f1:.2f} | Prec. = {test_prec:.2f} | Rec. = {test_rec:.2f}')

def get_lr_schedule(total_steps, warmup_ratio, hold_ratio, max_lr, min_lr):
    warmup_steps = int(total_steps * warmup_ratio)
    hold_steps = int(total_steps * hold_ratio)
    decay_steps = total_steps - warmup_steps - hold_steps
    min_delta = min_lr / max_lr

    def lr_schedule(current_step):
        if current_step < warmup_steps:
            # Warm-up (linear)
            return max((current_step / warmup_steps), min_delta)
        elif current_step < warmup_steps + hold_steps:
            # Hold
            return 1.0
        else:
            # Decay (2nd order polynomial)
            decay_progress = (current_step - warmup_steps - hold_steps) / decay_steps
            return max(((1 - decay_progress) ** 2), min_delta)

    return lr_schedule