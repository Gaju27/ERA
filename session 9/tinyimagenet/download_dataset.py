import os
from zipfile import ZipFile

# Make sure kaggle.json is present in ~/.kaggle or %USERPROFILE%\.kaggle
os.system("kaggle datasets download -d trolukovich/tiny-imagenet -p data --unzip")
print("Tiny ImageNet downloaded and extracted!")
