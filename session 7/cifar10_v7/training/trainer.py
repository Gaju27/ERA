 import torch
 import torch.nn as nn
 import torch.optim as optim
 from torch.utils.tensorboard import SummaryWriter
 from pathlib import Path

 from ..utils.metrics import AverageMeter, calculate_accuracy
 from ..utils.helpers import format_time
 from .early_stopping import EarlyStopping


 class Trainer:
     def __init__(self, model: nn.Module, config: dict, device: torch.device, logger):
         self.model = model
         self.cfg = config
         self.device = device
         self.logger = logger

         self.optimizer = self._create_optimizer()
         self.scheduler = self._create_scheduler()
         self.criterion = nn.CrossEntropyLoss()
         self.early_stopping = EarlyStopping(
             patience=config["early_stopping"]["patience"],
             min_delta=config["early_stopping"]["min_delta"],
         )

         self.ckpt_dir = Path(config["checkpoint_dir"]).resolve()
         self.ckpt_dir.mkdir(parents=True, exist_ok=True)

         self.writer = SummaryWriter(log_dir="runs/cifar10_v7")

     def _create_optimizer(self):
         name = self.cfg["optimizer"].lower()
         lr = self.cfg["learning_rate"]
         wd = self.cfg.get("weight_decay", 0.0)
         if name == "adam":
             return optim.Adam(self.model.parameters(), lr=lr, weight_decay=wd)
         if name == "sgd":
             return optim.SGD(self.model.parameters(), lr=lr, weight_decay=wd, momentum=0.9)
         if name == "adamw":
             return optim.AdamW(self.model.parameters(), lr=lr, weight_decay=wd)
         raise ValueError(f"Unsupported optimizer: {name}")

     def _create_scheduler(self):
         name = self.cfg.get("scheduler", "").lower()
         if name == "cosine":
             return optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.cfg["epochs"])
         if name == "step":
             return optim.lr_scheduler.StepLR(self.optimizer, step_size=30, gamma=0.1)
         if name == "plateau":
             return optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode="max", factor=0.5, patience=5)
         return None

     def train(self, train_loader, val_loader):
         best_val = 0.0
         for epoch in range(self.cfg["epochs"]):
             train_loss, train_acc = self._train_epoch(train_loader, epoch)
             val_loss, val_acc = self._validate_epoch(val_loader)

             if self.scheduler:
                 if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                     self.scheduler.step(val_acc)
                 else:
                     self.scheduler.step()

             self.writer.add_scalar("Loss/Train", train_loss, epoch)
             self.writer.add_scalar("Loss/Val", val_loss, epoch)
             self.writer.add_scalar("Acc/Train", train_acc, epoch)
             self.writer.add_scalar("Acc/Val", val_acc, epoch)

             self.logger.info(
                 f"Epoch {epoch+1}/{self.cfg['epochs']} - Train: {train_loss:.4f}/{train_acc:.4f} "
                 f"Val: {val_loss:.4f}/{val_acc:.4f}"
             )

             if val_acc > best_val:
                 best_val = val_acc
                 self._save_checkpoint(epoch, best=True, val_acc=val_acc)

             if self.early_stopping(val_acc, model=self.model):
                 self.logger.info("Early stopping triggered")
                 break

     def _train_epoch(self, loader, epoch):
         self.model.train()
         losses = AverageMeter()
         accs = AverageMeter()
         for idx, (x, y) in enumerate(loader):
             x, y = x.to(self.device), y.to(self.device)
             self.optimizer.zero_grad()
             out = self.model(x)
             loss = self.criterion(out, y)
             loss.backward()
             self.optimizer.step()
             acc = calculate_accuracy(out, y)
             losses.update(loss.item(), x.size(0))
             accs.update(acc, x.size(0))
             if idx % self.cfg["log_interval"] == 0:
                 self.logger.info(f"Epoch {epoch+1} [{idx}/{len(loader)}] loss={losses.avg:.4f} acc={accs.avg:.4f}")
         return losses.avg, accs.avg

     def _validate_epoch(self, loader):
         self.model.eval()
         losses = AverageMeter()
         accs = AverageMeter()
         with torch.no_grad():
             for x, y in loader:
                 x, y = x.to(self.device), y.to(self.device)
                 out = self.model(x)
                 loss = self.criterion(out, y)
                 acc = calculate_accuracy(out, y)
                 losses.update(loss.item(), x.size(0))
                 accs.update(acc, x.size(0))
         return losses.avg, accs.avg

     def test(self, loader):
         self.model.eval()
         correct = 0
         total = 0
         with torch.no_grad():
             for x, y in loader:
                 x, y = x.to(self.device), y.to(self.device)
                 out = self.model(x)
                 pred = out.argmax(dim=1)
                 correct += (pred == y).sum().item()
                 total += y.size(0)
         acc = correct / total
         self.logger.info(f"Test accuracy: {acc:.4f}")
         return acc

     def _save_checkpoint(self, epoch: int, best: bool, val_acc: float):
         ckpt = {
             "epoch": epoch,
             "model": self.model.state_dict(),
             "optimizer": self.optimizer.state_dict(),
             "val_acc": val_acc,
         }
         path = self.ckpt_dir / ("best.pth" if best else f"epoch_{epoch}.pth")
         torch.save(ckpt, path)

     def load_checkpoint(self, path: str):
         ckpt = torch.load(path, map_location=self.device)
         self.model.load_state_dict(ckpt["model"])
         self.optimizer.load_state_dict(ckpt["optimizer"])
         return ckpt.get("epoch", 0), ckpt.get("val_acc", 0.0)


