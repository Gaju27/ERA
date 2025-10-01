 import yaml
 from pathlib import Path


 class DotDict(dict):
     __getattr__ = dict.get
     __setattr__ = dict.__setitem__
     __delattr__ = dict.__delitem__


 class Config:
     def __init__(self, path: str):
         self.path = Path(path)
         with open(self.path, "r") as f:
             cfg = yaml.safe_load(f)
         # expose sections as attributes for convenience
         self._cfg = cfg
         self.data = DotDict(cfg.get("data", {}))
         self.model = DotDict(cfg.get("model", {}))
         self.training = DotDict(cfg.get("training", {}))
         self.logging = DotDict(cfg.get("logging", {}))

     def get(self, key, default=None):
         parts = key.split(".")
         curr = self._cfg
         for p in parts:
             curr = curr.get(p)
             if curr is None:
                 return default
         return curr


