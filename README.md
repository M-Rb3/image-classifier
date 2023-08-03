# Image Classifier

![Python](https://img.shields.io/badge/python-3.9-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## Description

Image Classifier is a Python-based deep learning project that uses a pre-trained convolutional neural network (CNN) to classify images. The model is built using PyTorch and is trained on a dataset of images from various categories. The trained model can then be used to predict the class of input images.

## Features

- Pre-trained CNN for image classification
- Supports multiple image formats
- Easy-to-use command-line interface
- Fast and efficient prediction

## Requirements

- Python 3.9 or higher
- PyTorch (torch) library
- TorchVision (torchvision) library
- NumPy (numpy) library

Install the required dependencies using the following command:

pip install torch torchvision numpy

markdown
Copy code

## Usage

1. Clone the repository:

git clone https://github.com/M-Rb3/image-classifier.git
cd image-classifier

less
Copy code

2. Download the pre-trained model weights:

Download the model weights file `model_weights.pth` from [Google Drive](https://drive.google.com/file/d/xyz1234567890/view?usp=sharing) and place it in the project root directory.

3. Classify an image:

python classify.py path/to/your/image.jpg

csharp
Copy code

Replace `path/to/your/image.jpg` with the path to the image you want to classify.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- The pre-trained model used in this project is based on [ResNet-50](https://arxiv.org/abs/1512.03385) architecture.
