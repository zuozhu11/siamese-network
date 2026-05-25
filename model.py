import torch.nn as nn
import torch.nn.functional as F

import torch
import numpy as np

import numpy as np
from PIL import Image

from torch.utils.data import Dataset
from torch.utils.data.sampler import BatchSampler

import torch
from torch.optim import lr_scheduler
import torch.optim as optim
from torch.autograd import Variable

import numpy as np
device = torch.cuda.is_available()

from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

# model definition
class embedding_net(nn.Module):
    def __init__(self):
        super().__init__()
        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1
        mobilenet = mobilenet_v3_small(weights=weights)

        self.features = mobilenet.features

        self.fc = nn.Sequential(
            nn.Linear(576, 256),
            nn.PReLU(),
            nn.Linear(256, 128)
        )

    def forward(self, x):
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
            
        output = self.features(x)
        output = F.adaptive_avg_pool2d(output, (1, 1))
        output = output.view(output.size()[0], -1)
        output = self.fc(output)
        return output
    
    def embedding(self, x):
        return self.forward(x)
    
class SiameseNet(nn.Module):
    def __init__(self, embedding_net):
        super().__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2):
        x1 = self.embedding_net(x1)
        x2 = self.embedding_net(x2)

        return x1, x2 
    
    def embedding(self, x):
        return self.embedding_net(x)

# dataset
class SiameseMNIST(Dataset):
    def __init__(self, mnist_dataset):
        self.mnist_dataset = mnist_dataset
        self.train = self.mnist_dataset.train
        self.transform = self.mnist_dataset.transform

        if self.train:
            self.train_labels = self.mnist_dataset.train_labels
            self.train_data = self.mnist_dataset.train_data
            self.label_set = set(self.train_labels.numpy())
            self.labels_to_indices = {label: np.where(self.train_labels.numpy() == label)[0]
                                      for label in self.label_set}
        else:
            self.test_labels = self.mnist_dataset.test_labels
            self.test_data = self.mnist_dataset.test_data
            self.label_set = set(self.test_labels.numpy())
            self.labels_to_indices = {label: np.where(self.test_labels.numpy() == label)[0]
                                     for label in self.label_set}
            
            random_state = np.random.RandomState(29)

            positive_set = [[i,
                            random_state.choice(self.labels_to_indices[self.test_labels[i].item()]),
                            1] for i in range(0, len(self.test_data), 2)]
            
            negative_set = [[i,
                             random_state.choice(self.labels_to_indices[np.random.choice(list(self.label_set - {self.test_labels[i].item()}))]),
                             0] for i in range(1, len(self.test_data), 2)]
            
            self.test_set = positive_set + negative_set

    def __getitem__(self, index):
        if self.train:
            img1, label1 = self.train_data[index], self.train_labels[index].item()

            target = np.random.randint(0,2)

            if target == 1:
                siamese_index = index 
                while siamese_index == index:
                    siamese_index = np.random.choice(self.labels_to_indices[label1])

            else:
                label2 = np.random.choice(list(self.label_set - {label1}))
                siamese_index = np.random.choice(self.labels_to_indices[label2])

            img2 = self.train_data[siamese_index]

        else:
            img1 = self.test_data[self.test_set[index][0]]
            img2 = self.test_data[self.test_set[index][1]]
            target = self.test_set[index][2]

        img1 = Image.fromarray(img1.numpy())
        img2 = Image.fromarray(img2.numpy())

        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        
        return (img1, img2), target
    
    def __len__(self):
        return len(self.mnist_dataset)

# loss function
class ContrastiveLoss(nn.Module):
    """
    Contrastive loss
    Takes embeddings of two samples and a target label == 1 if samples are from the same class and label == 0 otherwise
    """

    def __init__(self, margin):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.eps = 1e-9

    def forward(self, output1, output2, target):
        distances = (output2 - output1).pow(2).sum(1)  # squared distances
        losses = 0.5 * (target.float() * distances +
                        (1 + -1 * target).float() * F.relu(self.margin - (distances + self.eps).sqrt()).pow(2))
        return losses.mean()

# train loop and val loop
def train_epoch(train_loader, model, loss_fn, optimizer, device):
    model.train()
    total_loss = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data = tuple(d.to(device) for d in data)
        target = target.to(device)

        optimizer.zero_grad()

        outputs = model(*data)

        loss_inputs = outputs + (target,)

        loss = loss_fn(*loss_inputs)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / (batch_idx + 1)
    
def val_epoch(test_loader, model, loss_fn, device):
    with torch.no_grad():
        model.eval()
        total_loss = 0

        for batch_idx, (data, target) in enumerate(test_loader):
            data = tuple(d.to(device) for d in data)
            target = target.to(device)

            outputs = model(*data)

            loss_inputs = outputs + (target,)

            loss = loss_fn(*loss_inputs)

            total_loss += loss.item()

        avg_loss = total_loss / len(test_loader)
        # print(f'Validation set: Average loss: {avg_loss:.4f}')
        return avg_loss

def fit(train_loader, test_loader, model, loss_fn, optimizer, device, scheduler, n_epochs):
    best_val_loss = float('inf')
    for epoch in range(n_epochs):
        train_loss = train_epoch(train_loader, model, loss_fn, optimizer, device)
        val_loss = val_epoch(test_loader, model, loss_fn, device)
        
        print(f'Epoch: {epoch + 1}/{n_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            print(f'    New best model saved with validation loss: {best_val_loss:.4f}')

        scheduler.step()

import torch
from torchvision.datasets import FashionMNIST
from torchvision import transforms

# main
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mean, std = 0.28604059698879553, 0.35302424451492237
    batch_size = 256

    from torchvision.datasets import FashionMNIST
    train_dataset = FashionMNIST('../data/FashionMNIST', train=True, download=True,
                                transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize((mean,), (std,))
                                ]))
    test_dataset = FashionMNIST('../data/FashionMNIST', train=False, download=True,
                                transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize((mean,), (std,))
                                ]))

    cuda = torch.cuda.is_available()

    n_classes = 10

    siamese_train_dataset = SiameseMNIST(train_dataset) # Returns pairs of images and target same/different
    siamese_test_dataset = SiameseMNIST(test_dataset)
    batch_size = 128
    kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
    siamese_train_loader = torch.utils.data.DataLoader(siamese_train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
    siamese_test_loader = torch.utils.data.DataLoader(siamese_test_dataset, batch_size=batch_size, shuffle=False, **kwargs)

    model = SiameseNet(embedding_net()).to(device)

    loss_fn = ContrastiveLoss(margin=1.0)

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)

    n_epochs = 20
    fit(siamese_train_loader, siamese_test_loader, model, loss_fn, optimizer, device, scheduler, n_epochs)
