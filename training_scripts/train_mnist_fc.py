import os
import sys
sys.path.append(os.getcwd())
import vdp
import torch
from tqdm import tqdm
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader


class vdp_fc(torch.nn.Module):
    def __init__(self):
        super(vdp_fc, self).__init__()
        self.fc1 = vdp.Linear(28 * 28, 128, input_flag=True)
        self.fc2 = vdp.Linear(128, 64)
        self.fc3 = vdp.Linear(64, 10)
        self.relu = vdp.ReLU()
        self.softmax = vdp.Softmax()

    def forward(self, x):
        mu, sigma = self.fc1(x)
        mu, sigma = self.relu(mu, sigma)
        mu, sigma = self.fc2(mu, sigma)
        mu, sigma = self.relu(mu, sigma)
        mu, sigma = self.fc3(mu, sigma)
        mu, sigma = self.softmax(mu, sigma)
        return mu, sigma

    def get_loss(self, mu, sigma, y):
        log_det, nll = vdp.ELBOLoss(mu, sigma, y, num_classes=10, criterion='ce')
        kl = vdp.gather_kl(self)
        loss = log_det + 1000 * nll + sum(kl)
        return loss

    def score(self, logits, y):
        score = torch.sum(torch.argmax(logits, dim=1) == y)/len(logits)
        return score.cpu().numpy()


def train_model():
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    transform = transforms.Compose([transforms.ToTensor()]) #, transforms.Normalize((0.1307,), (0.3081,))])
    train = MNIST(os.getcwd(), train=True, download=True, transform=transform)
    test = MNIST(os.getcwd(), train=False, download=True, transform=transform)
    trainloader = DataLoader(train, batch_size=2048, num_workers=2,
                                shuffle=True,
                                pin_memory=True)  # IF YOU CAN FIT THE DATA INTO MEMORY DO NOT USE DATALOADERS
    testloader = DataLoader(test, batch_size=4096, num_workers=2, shuffle=True, pin_memory=True)
    model = vdp_fc()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, amsgrad=True)
    no_epochs = 50
    model.to(device)
    train_acc = 0
    for _ in tqdm(range(no_epochs)):
        for _, (x, labels) in tqdm(enumerate(trainloader), desc=f'Training Accuracy: {train_acc:.2f}', leave=False):
            optimizer.zero_grad()
            mu, sigma = model(x.float().reshape(-1, 28*28).to(device))

            loss = model.get_loss(mu, sigma, labels.to(device))
            loss.backward()
            optimizer.step()

            train_acc = model.score(mu, labels.to(device))
    test_logits = list()
    test_labels = list()     
    for _, (x, labels) in tqdm(enumerate(testloader)):
        mu, sigma = model(x.float().reshape(-1, 28*28).to(device))
        test_logits.append(mu)
        test_labels.append(labels)
    test_acc = model.score(torch.concat(test_logits), torch.concat(test_labels).to(device))
    print(f'Test acc: {test_acc}')
    print('Saving Model...')
    torch.save(model.state_dict(), "mnist_fc_ce.pt")
        
        
if __name__ == '__main__':
    train_model()
