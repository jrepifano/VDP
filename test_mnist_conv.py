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
        self.conv_1 = vdp.Conv2D(1, 32, 5, 1, input_flag=True)
        self.pool_1 = vdp.MaxPool2D(2, 2)
        self.conv_2 = vdp.Conv2D(32, 64, 5)
        self.pool_2 = vdp.MaxPool2D(2, 2)
        self.linear_1 = vdp.Linear(1024, 600)
        self.linear_2 = vdp.Linear(600, 10)
        self.relu = vdp.ReLU()
        self.softmax = vdp.Softmax()

    def forward(self, x):
        mu, sigma = self.conv_1(x)
        mu, sigma = self.relu(mu, sigma)
        mu, sigma = self.pool_1(mu, sigma)
        mu, sigma = self.conv_2(mu, sigma)
        mu, sigma = self.relu(mu, sigma)
        mu, sigma = self.pool_2(mu, sigma)
        mu, sigma = mu.view(mu.size(0), -1), sigma.view(mu.size(0), -1)
        mu, sigma = self.linear_1(mu, sigma)
        mu, sigma = self.relu(mu, sigma)
        mu, sigma = self.linear_2(mu, sigma)
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
    trainloader = DataLoader(mnist_train, batch_size=2000, num_workers=2,
                             shuffle=True, pin_memory=True)  # IF YOU CAN FIT THE DATA INTO MEMORY DO NOT USE DATALOADERS
    testloader = DataLoader(mnist_test, batch_size=2000, num_workers=2, shuffle=True, pin_memory=True)
    model = Model()
    no_epochs = 10
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, amsgrad=True)
    model.to('cuda:0')
    train_accs = []
    test_accs = []
    start_time = time.time()
    log_det, log_likelihood, kl = list(), list(), list()
    alpha = 0.001
    beta_1 = 0.001
    for epoch in range(no_epochs):
        total_loss = list()
        for itr, (image, labels) in enumerate(trainloader):
            model.train()
            optimizer.zero_grad()
            mu, sigma = model.forward(image.float().to('cuda:0'))
            log_det_i, log_likelihood_i = vdp.ELBOLoss(mu, sigma, labels.to('cuda:0'))
            kl_i = beta_1*(model.conv_1.kl_term() + model.conv_2.kl_term() + model.linear_1.kl_term() + model.linear_2.kl_term())
            loss = alpha * log_det_i + log_likelihood_i + (kl_i)
            total_loss.append(loss.item())
            loss.backward()
            optimizer.step()
            log_det.append(log_det_i.detach().cpu().numpy())
            log_likelihood.append(log_likelihood_i.detach().cpu().numpy())
            kl.append(kl_i.detach().cpu().numpy())
            train_acc = model.score(mu, labels.to('cuda:0'))
            train_accs.append(train_acc)
            # print('log det: {:.4f}'.format(alpha * log_det_i))
            # print('likelihood: {:.4f}'.format(log_likelihood_i))
            # print('kl: {:.4f}'.format(kl_i))
            print('Epoch {}/{}, itr {}/{}: Training Loss: {:.2f}'.format(epoch + 1, no_epochs, itr, (60000//2000), np.mean(total_loss)))
            print('Train Accuracy: {:.2f}'.format(train_acc))
        mu_1 = pd.DataFrame(model.conv_1.mu.weight.view(-1, 1).detach().cpu().numpy(), columns=['conv_1'])
        mu_2 = pd.DataFrame(model.conv_2.mu.weight.view(-1, 1).detach().cpu().numpy(), columns=['conv_2'])
        mu_3 = pd.DataFrame(model.linear_1.mu.weight.view(-1, 1).detach().cpu().numpy(), columns=['linear_1'])
        mu_4 = pd.DataFrame(model.linear_2.mu.weight.view(-1, 1).detach().cpu().numpy(), columns=['linear_2'])
        sns.kdeplot(mu_1['conv_1'], shade=True, label='conv_1')
        sns.kdeplot(mu_2['conv_2'], shade=True, label='conv_2')
        sns.kdeplot(mu_3['linear_1'], shade=True, label='linear_1')
        sns.kdeplot(mu_4['linear_2'], shade=True, label='linear_2')
        plt.legend()
        plt.title('Mu Epoch {}'.format(epoch))
        plt.savefig('conv_kde/mu_epoch_{}.png'.format(epoch))
        plt.show()
        plt.clf()
        sig_1 = pd.DataFrame(model.conv_1.sigma.weight.view(-1, 1).detach().cpu().numpy(), columns=['conv_1'])
        sig_2 = pd.DataFrame(model.conv_2.sigma.weight.view(-1, 1).detach().cpu().numpy(), columns=['conv_2'])
        sig_3 = pd.DataFrame(model.linear_1.sigma.weight.view(-1, 1).detach().cpu().numpy(), columns=['linear_1'])
        sig_4 = pd.DataFrame(model.linear_2.sigma.weight.view(-1, 1).detach().cpu().numpy(), columns=['linear_2'])
        sns.kdeplot(sig_1['conv_1'], shade=True, label='conv_1')
        sns.kdeplot(sig_2['conv_2'], shade=True, label='conv_2')
        sns.kdeplot(sig_3['linear_1'], shade=True, label='linear_1')
        sns.kdeplot(sig_4['linear_2'], shade=True, label='linear_2')
        plt.legend()
        plt.title('Sigma Epoch {}'.format(epoch))
        plt.savefig('conv_kde/sigma_epoch_{}.png'.format(epoch))
        plt.show()
        plt.clf()
    plt.plot(np.arange(no_epochs), train_accs, label='Train Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Fashion MNIST: Small Network VDP++')
    plt.legend()
    plt.show()
    # plt.savefig('variable_tau/learning_curve.png')
    # torch.save(model.state_dict(), 'vdp.pt')
    # plt.plot(np.arange(no_epochs), log_det, label='Log determinant')
    # plt.xlabel('Epoch')
    # plt.title('Fashion MNIST: Small Network VDP++')
    # plt.legend()
    # plt.show()
    # plt.plot(np.arange(no_epochs), likelihood, label='Likelihood')
    # plt.xlabel('Epoch')
    # plt.title('Fashion MNIST: Small Network VDP++')
    # plt.legend()
    # plt.show()
    # plt.plot(np.arange(no_epochs), kl, label='KL terms')
    # plt.xlabel('Epoch')
    # plt.title('Fashion MNIST: Small Network VDP++')
    # plt.legend()
    # plt.show()
    # print('Train and val time for {} epochs: {:.2f}'.format(no_epochs, (time.time()-start_time)))
    # snrs = [-6, -3, 1, 5, 10, 20]
    # sigmas = list()
    # for snr in range(len(snrs)):
    #     x_noisy = add_noise(x_test.cpu().numpy(), snrs[snr])
    #     mu, sigma = model.forward(torch.from_numpy(x_noisy).float().to('cuda:0'))
    #     sigmas.append(torch.mean(torch.mean(sigma, dim=1)).detach().cpu().numpy())
    # plt.figure()
    # plt.plot(snrs, sigmas)
    # plt.xlabel('SNR (dB)')
    # plt.ylabel('Mean Mean Test Sigma')
    # plt.title('Fashion MNIST: Small network VDP++')
    # plt.show()
    # print(init_norms_mu)
    # print([torch.norm(model.layer_1.mu.weight), torch.norm(model.layer_2.mu.weight), torch.norm(model.layer_3.mu.weight)])
    # print(init_norms)
    # print([torch.norm(model.layer_1.sigma.weight), torch.norm(model.layer_2.sigma.weight), torch.norm(model.layer_3.sigma.weight)])
    pass


if __name__ == '__main__':
    main()
