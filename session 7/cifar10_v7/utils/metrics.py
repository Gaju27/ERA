 import torch


 class AverageMeter:
     def __init__(self):
         self.reset()

     def reset(self):
         self.val = 0
         self.avg = 0
         self.sum = 0
         self.count = 0

     def update(self, val: float, n: int = 1):
         self.val = val
         self.sum += val * n
         self.count += n
         self.avg = self.sum / self.count


 def calculate_accuracy(output: torch.Tensor, target: torch.Tensor) -> float:
     with torch.no_grad():
         pred = output.argmax(dim=1)
         correct = (pred == target).float().sum()
     return (correct / target.size(0)).item()


