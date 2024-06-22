#IMPORTS
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

#Create neural network
class CNN(nn.Module):
    def __init__(self,input_channel=1,num_classes=10):
        super(CNN,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=input_channel,out_channels=8,kernel_size=3,stride=1,padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2,stride=2)
        self.conv2 = nn.Conv2d(in_channels=8,out_channels=16,kernel_size=3,stride=1,padding=1)
        self.fc1 = nn.Linear(16*7*7, num_classes)

    def forward(self, x):
        x=F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)

        return x
    
#Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Hyperparameters
lr = 0.001
batch_size  = 64
num_epochs = 5
input_channel = 1
num_classes = 10

#dataste
train_dataset = datasets.MNIST(root='dataset/' ,train=True, transform=transforms.ToTensor(),download=True)
train_loader = DataLoader(dataset = train_dataset, batch_size = batch_size , shuffle = True)
test_dataset = datasets.MNIST(root='dataset/' ,train=False, transform=transforms.ToTensor(),download=True)
test_loader = DataLoader(dataset = test_dataset, batch_size = batch_size , shuffle = True)

#Initialization
model = CNN(input_channel=input_channel,num_classes=num_classes).to(device)

# Test
# x = torch.randn(64,1,28,28)
# print(model(x).shape)
# exit()

#Loss
criteria = nn.CrossEntropyLoss()
optimizer = optim.Adam(params=model.parameters(), lr = lr)

#model
for epoch in range(num_epochs):
    for idx ,(image, target) in enumerate(train_loader):
        image = image.to(device)
        target = target.to(device)

        output = model(image)
        loss = criteria(output, target)

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

#Accuracy
def chech_accuracy(model,test_loader):
    num_correct = 0
    num_samples = 0
    model.eval()
    with torch.no_grad():
        for x,y in test_loader:
            x = x.to(device)
            y = y.to(device)

            scores = model(x)
            _,pred = scores.max(1)
            num_correct += (pred == y).sum()
            num_samples += pred.size(0)

        print(f"accuracy is {num_correct/num_samples*100}")

    model.train()


chech_accuracy(model,train_loader)
chech_accuracy(model,test_loader)
   



