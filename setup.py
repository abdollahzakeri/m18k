from setuptools import setup, find_packages

with open("README.md", 'r') as f:
    long_description = f.read()

setup(
   name='M18K',
   version='1.0',
   description='Mushroom RGB-D image dataset for object detection and instance segmentation',
   license="MIT",
   long_description=long_description,
   author='Abdollah Zakeri',
   author_email='azakeri@uh.edu',
   url="http://www.abdollahzakeri.github.io/",
   packages=find_packages(),  #same as name
   install_requires=['opencv-python', 'matplotlib', 'ultralytics', "torch", "albumentations", "pycocotools", "lightning", "timm", "scipy", "torchvision", "gdown","jupyter"], #external packages as dependencies
)