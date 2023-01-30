import os
import sys
sys.path.append(os.getcwd())
import vdp
import torch
from tqdm import tqdm
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader


class vdp_lenet(torch.nn.Module):
    def __init__(self):
        super(vdp_lenet, self).__init__()
        self.conv1 = vdp.Conv2d(1, 6, 5, padding=2, input_flag=True)
        self.conv2 = vdp.Conv2d(6, 16, 5)
        self.conv3 = vdp.Conv2d(16, 120, 5)
        self.fc1 = vdp.Linear(120, 84)  # 5*5 from image dimension
        self.pool = vdp.MaxPool2d(2, 2)
        self.relu = vdp.ReLU()
        self.lin_last = vdp.Linear(84, 10)
        self.softmax = vdp.Softmax()


    def forward(self, x):
        mu, sigma = self.conv1(x)
        mu, sigma = self.relu(mu, sigma)
        mu, sigma = self.pool(mu, sigma)

        mu, sigma = self.conv2(mu, sigma)
        mu, sigma = self.relu(mu, sigma)
        mu, sigma = self.pool(mu, sigma)

        mu, sigma = self.conv3(mu, sigma)
        mu, sigma = self.relu(mu, sigma)

        mu = torch.flatten(mu, 1)
        sigma = torch.flatten(sigma, 1)

        mu, sigma = self.fc1(mu, sigma)
        mu, sigma = self.relu(mu, sigma)

        mu, sigma = self.lin_last(mu, sigma)
        mu, sigma = self.softmax(mu, sigma)
        return mu, sigma


    def get_loss(self, mu, sigma, y):
        log_det, nll = vdp.ELBOLoss(mu, sigma, y)
        kl = vdp.gather_kl(self)
        # if self.alpha is None:
        #     self.alpha, self.tau = vdp.scale_hyperp(log_det, nll, kl)
        # loss = self.alpha * log_det + nll + self.tau * sum(kl)
        loss = log_det + 100 * nll + sum(kl)
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
    # test = MNIST(os.getcwd(), train=False, download=True, transform=transform)
    trainloader = DataLoader(train, batch_size=4096, num_workers=2,
                                shuffle=True,
                                pin_memory=True)  # IF YOU CAN FIT THE DATA INTO MEMORY DO NOT USE DATALOADERS
    # testloader = DataLoader(test, batch_size=4096, num_workers=2, shuffle=True, pin_memory=True)
    model = vdp_lenet()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, amsgrad=True)
    no_epochs = 50
    model.to(device)
    train_acc = 0
    for _ in tqdm(range(no_epochs)):
        for _, (x, labels) in tqdm(enumerate(trainloader), desc=f'Training Accuracy: {train_acc:.2f}', leave=False):
            optimizer.zero_grad()
            mu, sigma = model(x.float().to(device))

            loss = model.get_loss(mu, sigma, labels.to(device))
            loss.backward()
            optimizer.step()

            train_acc = model.score(mu, labels.to(device))         

    # print('Saving Model...')
    # torch.save(self.state_dict(), dir + ".pt")
        
        
if __name__ == '__main__':
    train_model()
