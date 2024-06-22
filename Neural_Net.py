#Imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

#Create neural net
class NN(nn.Module):
    def __init__(self,input_size,num_classes):
        super(NN,self).__init__()
        self.fc1 = nn.Linear(input_size, 500)
        self.fc2 = nn.Linear(500,100)
        self.fc3 = nn.Linear(100,num_classes)
        self.dropout = nn.Dropout(p=0.5) 

    def forward(self ,x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x= self.fc3(x)
        return x

#Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Hypermeters 
input_size = 784
num_classes = 10
lr = 0.001
batch_size  = 64
num_epochs = 5

#Load Data
train_dataset = datasets.MNIST(root='dataset/' ,train=True, transform=transforms.ToTensor(),download=True)
train_loader = DataLoader(dataset = train_dataset, batch_size = batch_size , shuffle = True)
test_dataset = datasets.MNIST(root='dataset/' ,train=False, transform=transforms.ToTensor(),download=True)
test_loader = DataLoader(dataset = test_dataset, batch_size = batch_size , shuffle = True)

# Initialize
model = NN(input_size=input_size,num_classes=num_classes).to(device)
  
#Loss & Optim
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(params=model.parameters(),lr=lr)

# Train Network

for epoch in range(num_epochs):
    for idx ,(image, target) in enumerate(train_loader):
        image = image.to(device)
        target = target.to(device)

        image = image.reshape(image.shape[0],-1)

        output = model(image)
        loss = criterion(output,target)

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()


def chech_accuracy(model,test_loader):
    num_correct = 0
    num_samples = 0
    model.eval()
    with torch.no_grad():
        for x,y in test_loader:
            x = x.to(device)
            y = y.to(device)
            x = x.reshape(x.shape[0],-1)

            scores = model(x)
            _,pred = scores.max(1)
            num_correct += (pred == y).sum()
            num_samples += pred.size(0)

        print(f"accuracy is {num_correct/num_samples*100}")

    model.train()


chech_accuracy(model,train_loader)
chech_accuracy(model,test_loader)