 class EarlyStopping:
     def __init__(self, patience: int = 10, min_delta: float = 0.0):
         self.patience = patience
         self.min_delta = min_delta
         self.best = None
         self.count = 0
         self.early_stop = False
         self.best_weights = None

     def __call__(self, metric: float, model=None) -> bool:
         if self.best is None or metric > self.best + self.min_delta:
             self.best = metric
             self.count = 0
             if model is not None:
                 self.best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
         else:
             self.count += 1
             if self.count >= self.patience:
                 self.early_stop = True
                 if model is not None and self.best_weights is not None:
                     model.load_state_dict(self.best_weights)
         return self.early_stop

     def reset(self):
         self.best = None
         self.count = 0
         self.early_stop = False
         self.best_weights = None


