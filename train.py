import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from workspace_utils import active_session
from collections import OrderedDict 
import numpy as np
import os
import argparse
from operator import itemgetter
from utils import load_data
parser = argparse.ArgumentParser(description='network parameters')

# If I use data_directory, it won't be optional and the it will make data_directory as the arg value.
parser.add_argument('--data_directory', action="store", default='./flowers')
parser.add_argument('--save_dir', action="store", default="checkpoint.pth")
parser.add_argument('--arch', action="store",default="vgg19")
parser.add_argument('--learning_rate', action="store",default=0.001, type=int)
parser.add_argument('--hidden_units', action="store",default=512, type=int)
parser.add_argument('--epochs', action="store",default=4, type=int)
parser.add_argument('--gpu', action="store",default=True)


args = parser.parse_args()
arch, data_directory, epochs, gpu,hidden_units,learning_rate,save_dir = itemgetter('arch', 'data_directory', 'epochs', 'gpu','hidden_units','learning_rate','save_dir')(vars(args))
dataloaders,image_datasets=load_data(data_directory)
trainloader,validloader,testloader=itemgetter('trainloader', 'validloader', 'testloader')(dataloaders)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not gpu:
        device = torch.device("cpu")
    if arch=='vgg19':
        model = models.vgg19(pretrained=True)
    else:
        model = getattr(torchvision.models, arch)(pretrained=True)
        
    for param in model.parameters():
        param.requires_grad = False
    
    model.classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(25088, hidden_units)),
                          ('relu', nn.ReLU()),
                          ('fc2', nn.Linear(hidden_units, 256)),
                          ('relu', nn.ReLU()),
                          ('fc3', nn.Linear(256, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))    
    criterion = nn.NLLLoss()

    # Only train the classifier parameters, feature parameters are frozen
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

    model.to(device)
    
    epochs =epochs
    steps = 0
    running_loss = 0
    print_every = 5
    for epoch in range(epochs):
        for inputs, labels in trainloader:
            steps += 1
            # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                test_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in testloader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)

                        test_loss += batch_loss.item()

                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Test loss: {test_loss/len(testloader):.3f}.. "
                      f"Test accuracy: {accuracy/len(testloader):.3f}")
                running_loss = 0
                model.train()
    
    # Save the checkpoint
    os.remove("checkpoint.pth")
    checkpoint = {'input_size': 25088,
                  'output_size': 102,
                  'classifier': model.classifier,
                  'state_dict': model.state_dict(),
                  'optimizer':optimizer.state_dict,
                  'epochs':epochs,
                  'learning_rate':0.001,
                 'class_to_idx':image_datasets['train_data'].class_to_idx}
    
    torch.save(checkpoint, 'checkpoint.pth')
    
    
    
if __name__ == "__main__":
    main()