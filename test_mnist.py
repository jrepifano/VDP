import os
import vdp
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.datasets import FashionMNIST
from torch.optim.lr_scheduler import StepLR
import pandas as pd
import seaborn as sns
from torch.utils.data import DataLoader


os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # mnist images are (1, 28, 28) (channels, width, height)
        self.layer_1 = vdp.Linear(28 * 28, 128, input_flag=True)
        self.layer_2 = vdp.Linear(128, 50)
        self.layer_3 = vdp.Linear(50, 10)
        self.bn1 = torch.nn.BatchNorm1d(128)
        self.bn2 = torch.nn.BatchNorm1d(50)
        self.bn3 = torch.nn.BatchNorm1d(10)
        self.relu = vdp.ReLU()
        self.softmax = vdp.Softmax()

    def forward(self, x):
        # x = x.view(1000, -1)
        mu, sigma = self.layer_1(x)
        # print(mu)
        # mu = self.bn1(mu)
        mu, sigma = self.relu(mu, sigma)
        # print(mu)
        mu, sigma = self.layer_2(mu, sigma)
        # print(mu)
        # mu = self.bn2(mu)
        mu, sigma = self.relu(mu, sigma)
        # print(mu)
        mu, sigma = self.layer_3(mu, sigma)
        # print(mu)
        # mu = self.bn3(mu)
        mu, sigma = self.softmax(mu, sigma)
        # print(mu)
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
    # transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    mnist_train = FashionMNIST(os.getcwd(), train=True, download=True, transform=None)
    mnist_test = FashionMNIST(os.getcwd(), train=False, download=True, transform=None)
    # trainloader = DataLoader(mnist_train, batch_size=1000, num_workers=2,
    #                          shuffle=True)  # IF YOU CAN FIT THE DATA INTO MEMORY DO NOT USE DATALOADERS
    # testloader = DataLoader(mnist_test, batch_size=1000, num_workers=2, shuffle=True)
    x_train, y_train = mnist_train.data.view(60000, -1)/255.0, mnist_train.targets
    x_test, y_test = mnist_test.data.view(10000, -1)/255.0, mnist_test.targets
    model = Model()
    no_epochs = 200
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, amsgrad=True)
    model.to('cuda:0')
    train_accs = []
    test_accs = []
    start_time = time.time()
    log_det, likelihood, kl = list(), list(), list()
    for epoch in range(no_epochs):
        # for itr, (x_train, y_train) in enumerate(trainloader):
        model.train()
        total_loss = 0
        optimizer.zero_grad()
        mu, sigma = model.forward(x_train.float().to('cuda:0'))
        log_det_i, likelihood_i = vdp.ELBOLoss(mu, sigma, y_train.to('cuda:0'))
        kl_i = (model.layer_1.kl_term()+model.layer_2.kl_term()+model.layer_3.kl_term())
        loss = 1*log_det_i+likelihood_i-0.001*(kl_i)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        log_det.append(log_det_i.detach().cpu().numpy())
        likelihood.append(likelihood_i.detach().cpu().numpy())
        kl.append(kl_i.detach().cpu().numpy())
        train_acc = model.score(mu, y_train.to('cuda:0'))
        print('Epoch {}/{}: Training Loss: {:.2f}'.format(epoch + 1, no_epochs, total_loss))
        print('Train Accuracy: {:.2f}'.format(train_acc))
        train_accs.append(train_acc)
        # if (epoch % 25) == 0 or (epoch == 49):
        #     sns.kdeplot(pd.DataFrame(model.layer_1.mu.weight.view(-1, 1).detach().cpu().numpy(), columns=['mu'])[
        #                     'mu'], shade=True)
        #     plt.savefig('plots/mu_weights_epoch_{}.png'.format(epoch))
        #     plt.clf()
        #     sns.kdeplot(pd.DataFrame(model.layer_1.sigma.weight.view(-1, 1).detach().cpu().numpy(), columns=['sigma'])[
        #                     'sigma'], shade=True)
        #     plt.savefig('plots/sigma_weights_epoch_{}.png'.format(epoch))
        #     plt.clf()
        #     sns.kdeplot(pd.DataFrame(model.layer_1.mu.weight.grad.view(-1, 1).detach().cpu().numpy(), columns=['mu gradient'])[
        #                     'mu gradient'], shade=True)
        #     plt.savefig('plots/mu_grad_epoch_{}.png'.format(epoch))
        #     plt.clf()
        #     sns.kdeplot(pd.DataFrame(model.layer_1.sigma.weight.grad.view(-1, 1).detach().cpu().numpy(), columns=['sigma gradient'])[
        #                     'sigma gradient'], shade=True)
        #     plt.savefig('plots/sigma_grad_epoch_{}.png'.format(epoch))
        #     plt.clf()
        # print(model.layer_1.mu.weight.grad)
        # print(model.layer_1.sigma.weight.grad)
        # for itr, (x_test, y_test) in enumerate(trainloader):
        model.eval()  # This removes stuff like dropout and batch norm for inference stuff
        total_loss = 0
        mu, sigma = model.forward(x_test.float().to('cuda:0'))
        log_det_i, likelihood_i = vdp.ELBOLoss(mu, sigma, y_test.to('cuda:0'))
        kl_i = (model.layer_1.kl_term()+model.layer_2.kl_term()+model.layer_3.kl_term())
        loss = log_det_i+likelihood_i+kl_i
        total_loss += loss.item()
        test_acc = model.score(mu, y_test.to('cuda:0'))
        print('Test Loss: {:.2f},    Test Accuracy: {:.2f}'.format(total_loss, test_acc))
        test_accs.append(test_acc)
    plt.plot(np.arange(no_epochs), train_accs, label='Train Accuracy')
    plt.plot(np.arange(no_epochs), test_accs, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Fashion MNIST: Small Network VDP++')
    plt.legend()
    plt.show()
    plt.plot(np.arange(no_epochs), log_det, label='Log determinant')
    plt.xlabel('Epoch')
    plt.title('Fashion MNIST: Small Network VDP++')
    plt.legend()
    plt.show()
    plt.plot(np.arange(no_epochs), likelihood, label='Likelihood')
    plt.xlabel('Epoch')
    plt.title('Fashion MNIST: Small Network VDP++')
    plt.legend()
    plt.show()
    plt.plot(np.arange(no_epochs), kl, label='KL terms')
    plt.xlabel('Epoch')
    plt.title('Fashion MNIST: Small Network VDP++')
    plt.legend()
    plt.show()
    print('Train and val time for {} epochs: {:.2f}'.format(no_epochs, (time.time()-start_time)))
    snrs = [-6, -3, 1, 5, 10, 20]
    sigmas = list()
    for snr in range(len(snrs)):
        x_noisy = add_noise(x_test.cpu().numpy(), snrs[snr])
        mu, sigma = model.forward(torch.from_numpy(x_noisy).float().to('cuda:0'))
        sigmas.append(torch.mean(torch.mean(sigma, dim=1)).detach().cpu().numpy())
    plt.figure()
    plt.plot(snrs, sigmas)
    plt.xlabel('SNR (dB)')
    plt.ylabel('Mean Mean Test Sigma')
    plt.title('Fashion MNIST: Small network VDP++')
    plt.show()
    # print(init_norms_mu)
    # print([torch.norm(model.layer_1.mu.weight), torch.norm(model.layer_2.mu.weight), torch.norm(model.layer_3.mu.weight)])
    # print(init_norms)
    # print([torch.norm(model.layer_1.sigma.weight), torch.norm(model.layer_2.sigma.weight), torch.norm(model.layer_3.sigma.weight)])
    pass



if __name__ == '__main__':
    main()
