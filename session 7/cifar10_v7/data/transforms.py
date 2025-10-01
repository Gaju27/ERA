 from torchvision import transforms


 def get_transforms(config: dict, is_training: bool = True) -> transforms.Compose:
     size = config["image_size"]
     mean = config["normalize"]["mean"]
     std = config["normalize"]["std"]

     if is_training:
         aug = [
             transforms.RandomCrop(size, padding=4),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             transforms.Normalize(mean=mean, std=std),
         ]
     else:
         aug = [
             transforms.ToTensor(),
             transforms.Normalize(mean=mean, std=std),
         ]

     return transforms.Compose(aug)


