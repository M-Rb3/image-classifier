import torch
import json
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from workspace_utils import active_session
from collections import OrderedDict 
from PIL import Image
import numpy as np
import os
import argparse
from operator import itemgetter
from utils import load_data,process_image,imshow

parser = argparse.ArgumentParser(description='network parameters')

# If I use data_directory, it won't be optional and the it will make data_directory as the arg value.
parser.add_argument('--img_path', action="store", default='./flowers/test/1/image_06743.jpg')  
parser.add_argument('--top_k', action="store",default=5, type=int)
parser.add_argument('--checkpoint', action="store",default='./checkpoint.pth')
parser.add_argument('--category_names', action="store",default='./cat_to_name.json')
parser.add_argument('--gpu', action="store",default=True)

args = parser.parse_args()
img_path, top_k, checkpoint, category_names,gpu = itemgetter('img_path', 'top_k', 'checkpoint', 'category_names','gpu')(vars(args))

with open(category_names, 'r') as f:
    cat_to_name = json.load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if not gpu:
    device = torch.device("cpu")
    
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = models.vgg19(pretrained=True)
    
    for param in model.parameters():
        param.requires_grad = False
    criterion = nn.NLLLoss()
    # Only train the classifier parameters, feature parameters are frozen
#     optimizer =optim(checkpoint['optimizer'])
    model.to(device)
    
    model.classifier = checkpoint['classifier']
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])
    
    return model

def main():
    model = load_checkpoint(checkpoint)
    
    model.to(device)
    model.eval()
    im = process_image(img_path)
    im =  torch.from_numpy(im).float()
    im.unsqueeze_(0) 
    im = im.to(device)
    with torch.no_grad():
        logps = model.forward(im)
        
    ps = torch.exp(logps)
    
    probabilities, classes = ps.topk(top_k, dim=1)
    labels = [cat_to_name[str(index+1)] for index in np.array(classes[0])]
    probabilities=np.array(probabilities[0])
    
    print('classes and probabilities:- ',list(zip(labels,probabilities)))
    print(f"Image class with highes probabilty is {labels[0]} and has probability of {round((probabilities[0]*100),2)}% ")
    return probabilities,classes
    
if __name__ == "__main__":
    main()
