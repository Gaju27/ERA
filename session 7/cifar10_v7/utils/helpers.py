 import torch
 import random
 import numpy as np


 def set_seed(seed: int = 42):
     random.seed(seed)
     np.random.seed(seed)
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     torch.backends.cudnn.deterministic = True
     torch.backends.cudnn.benchmark = False

 def format_time(seconds: float) -> str:
     if seconds < 60:
         return f"{seconds:.2f}s"
     if seconds < 3600:
         m = int(seconds // 60)
         s = seconds % 60
         return f"{m}m {s:.2f}s"
     h = int(seconds // 3600)
     m = int((seconds % 3600) // 60)
     s = seconds % 60
     return f"{h}h {m}m {s:.2f}s"


