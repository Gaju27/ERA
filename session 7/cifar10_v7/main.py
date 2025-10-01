 """
 Entry point for CIFAR10 v7 modular project (kept separate).
 """

 import argparse
 from pathlib import Path
 import torch

 from config.config import Config
 from data.data_loader import get_data_loaders
 from models.model_factory import create_model
 from training.trainer import Trainer
 from utils.logger import setup_logger
 from utils.helpers import set_seed


 def main():
     parser = argparse.ArgumentParser(description="CIFAR10 v7 Training/Testing")
     parser.add_argument("--config", type=str, default="config/config.yaml")
     parser.add_argument("--mode", type=str, choices=["train", "test"], default="train")
     parser.add_argument("--checkpoint", type=str, default=None)
     parser.add_argument("--seed", type=int, default=42)
     args = parser.parse_args()

     set_seed(args.seed)

     cfg = Config(args.config)
     logger = setup_logger(cfg.logging.log_level)
     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
     logger.info(f"Using device: {device}")

     train_loader, val_loader, test_loader = get_data_loaders(cfg.data)
     model = create_model(cfg.model).to(device)

     trainer = Trainer(model, cfg.training, device, logger)

     if args.mode == "train":
         trainer.train(train_loader, val_loader)
     else:
         if not args.checkpoint:
             logger.error("--checkpoint is required for test mode")
             return
         trainer.load_checkpoint(args.checkpoint)
         trainer.test(test_loader)


 if __name__ == "__main__":
     main()


