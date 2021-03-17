import os
import vdp
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.datasets import FashionMNIST
from torch.utils.data import DataLoader

os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'  # Set this flag to set your devices. For example if I set '6,7', then cuda:0 and cuda:1 in code will be cuda:6 and cuda:7 on hardware


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # mnist images are (1, 28, 28) (channels, width, height)
        self.layer_1 = vdp.Linear(28 * 28, 128, input_flag=True)
        self.layer_2 = vdp.Linear(128, 10)
        self.bn1 = torch.nn.BatchNorm1d(128)
        self.bn2 = torch.nn.BatchNorm1d(10)
        self.relu = vdp.ReLU()
        self.softmax = vdp.Softmax()

    def forward(self, x):
        batch_size, channels, width, height = x.size()
        # (b, 1, 28, 28) -> (b, 1*28*28)
        x = x.view(batch_size, -1)  # Reshapes image to 1-D tensor
        mu, sigma = self.layer_1(x)
        mu = self.bn1(mu)
        mu, sigma = self.relu(mu, sigma)
        mu, sigma = self.layer_2(mu, sigma)
        mu = self.bn2(mu)
        mu, sigma = self.softmax(mu, sigma)
        return mu, sigma

    def score(self, logits, y):
        score = torch.sum(torch.argmax(logits, dim=1) == y)/len(logits)
        return score.cpu().numpy()


def add_noise(s, snr):
    var_s = np.var(s, axis=1)
    var_n = var_s / (10 ** (snr / 10))
    rand_arr = np.random.randn(s.shape[0], s.shape[1])
    n = np.sqrt(var_n).reshape((-1, 1)) * rand_arr
    return s + n


def main():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    mnist_train = FashionMNIST(os.getcwd(), train=True, download=True, transform=transform)
    mnist_test = FashionMNIST(os.getcwd(), train=False, download=True, transform=transform)
    trainloader = DataLoader(mnist_train, batch_size=60000, num_workers=2,
                             shuffle=True)  # IF YOU CAN FIT THE DATA INTO MEMORY DO NOT USE DATALOADERS
    testloader = DataLoader(mnist_test, batch_size=10000, num_workers=2,
                            shuffle=True)  # Code will run so much faster without dataloaders for small(ish) datasets
    model = Model()
    no_epochs = 20
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    model.to('cuda:0')
    train_accs = []
    test_accs = []
    for epoch in range(no_epochs):
        model.train()
        total_loss = 0
        for itr, (x, y) in enumerate(trainloader):
            x, y = x.to('cuda:0'), y.to('cuda:0')
            optimizer.zero_grad()
            mu, sigma = model.forward(x)
            loss = vdp.ELBOLoss(mu, sigma, y)+0.002*(model.layer_1.kl_term()+model.layer_2.kl_term())
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            train_acc = model.score(mu, y)
        print('Epoch {}/{}: Training Loss: {:2f}'.format(epoch + 1, no_epochs, total_loss))
        print('Train Accuracy: {:.2f}'.format(train_acc))
        train_accs.append(train_acc)
        model.eval()  # This removes stuff like dropout and batch norm for inference stuff
        total_loss = 0
        for itr, (x, y) in enumerate(testloader):
            x, y = x.to('cuda:0'), y.to('cuda:0')
            mu, sigma = model.forward(x)
            loss = vdp.ELBOLoss(mu, sigma, y)+0.002*(model.layer_1.kl_term()+model.layer_2.kl_term())
            total_loss += loss.item()
            test_acc = model.score(mu, y)
        print('Test Loss: {:2f},    Test Accuracy: {:.2f}'.format(total_loss, test_acc))
        test_accs.append(test_acc)
    plt.plot(np.arange(no_epochs), train_accs, label='Train Accuracy')
    plt.plot(np.arange(no_epochs), test_accs, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Fashion MNIST: Small Network VDP++')
    plt.legend()
    plt.show()
    snrs = [-6, -3, 1, 5, 10, 20]
    sigmas = list()
    for snr in range(len(snrs)):
        for itr, (x, y) in enumerate(testloader):
            x, y = x.to('cuda:0'), y.to('cuda:0')
            x = add_noise(x.view(10000, -1).cpu().numpy(), snrs[snr])
            mu, sigma = model.forward(torch.from_numpy(x).view(10000, 1, 28, 28).float().to('cuda:0'))
            sigmas.append(torch.median(torch.mean(sigma, dim=1)).detach().cpu().numpy())
    plt.figure()
    plt.plot(snrs, sigmas)
    plt.xlabel('SNR (dB)')
    plt.ylabel('Median Mean Test Sigma')
    plt.title('Fashion MNIST: Small network VDP++')
    plt.show()
    pass


if __name__ == '__main__':
    main()
