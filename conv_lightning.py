import os
import vdp
import torch
import pandas as pd
import seaborn as sns
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import FashionMNIST, CIFAR10


class ConvNet(pl.LightningModule):
    def __init__(self, batch_size):
        super(ConvNet, self).__init__()
        self.batch_size = batch_size
        self.conv_1 = vdp.Conv2D(3, 64, 3, input_flag=True)
        self.conv_2 = vdp.Conv2D(64, 64, 3)
        self.conv_3 = vdp.Conv2D(64, 128, 3)
        self.conv_4 = vdp.Conv2D(128, 128, 3)
        self.pool = vdp.MaxPool2D(2, 2, padding=1)
        self.fc_1 = vdp.Linear(4608, 256)
        self.fc_2 = vdp.Linear(256, 256)
        self.fc_3 = vdp.Linear(256, 10)

        self.relu = vdp.ReLU()
        self.softmax = vdp.Softmax()
        self.dropout = torch.nn.Dropout(0.2)

        self.alpha = 0.001
        self.beta = 0.01
        self.epoch_counter = 0

        self.train_acc_step = pl.metrics.Accuracy()
        self.train_acc_epoch = pl.metrics.Accuracy()
        self.test_acc_step = pl.metrics.Accuracy()
        self.test_acc_epoch = pl.metrics.Accuracy()

    def forward(self, x):
        mu, sigma = self.conv_1(x)
        mu, sigma = self.relu(mu, sigma)

        mu, sigma = self.conv_2(mu, sigma)
        mu, sigma = self.relu(mu, sigma)
        mu, sigma = self.pool(mu, sigma)

        mu, sigma = self.conv_3(mu, sigma)
        mu, sigma = self.relu(mu, sigma)

        mu, sigma = self.conv_4(mu, sigma)
        mu, sigma = self.relu(mu, sigma)
        mu, sigma = self.pool(mu, sigma)

        mu, sigma = mu.view(mu.size(0), -1), sigma.view(mu.size(0), -1)

        mu, sigma = self.fc_1(mu, sigma)
        mu = self.dropout(mu)
        mu, sigma = self.relu(mu, sigma)

        mu, sigma = self.fc_2(mu, sigma)
        mu = self.dropout(mu)
        mu, sigma = self.relu(mu, sigma)

        mu, sigma = self.fc_3(mu, sigma)
        mu, sigma = self.relu(mu, sigma)

        mu, sigma = self.softmax(mu, sigma)

        return mu, sigma

    def training_step(self, batch, batch_idx):
        x, y = batch
        mu, sigma = self(x)
        log_det_i, nll_i = vdp.ELBOLoss(mu, sigma, y.to('cuda:0'))
        kl_i = self.beta * (self.conv_1.kl_term() + self.conv_2.kl_term() + self.conv_3.kl_term()
                            + self.conv_4.kl_term() + self.fc_1.kl_term() + self.fc_2.kl_term() + self.fc_3.kl_term())
        loss = self.alpha * log_det_i + nll_i + kl_i
        self.train_acc_epoch(mu, y)
        # self.log('train_acc_step', self.train_acc_step(mu, y), on_step=True, on_epoch=False)
        self.log('train_loss', loss.item(), prog_bar=False)
        return loss

    @torch.enable_grad()
    def validation_step(self, batch, batch_idx):
        x, y = batch
        mu, sigma = self(x)
        log_det_i, nll_i = vdp.ELBOLoss(mu, sigma, y.to('cuda:0'))
        kl_i = self.beta * (self.conv_1.kl_term() + self.conv_2.kl_term() + self.conv_3.kl_term()
                            + self.conv_4.kl_term() + self.fc_1.kl_term() + self.fc_2.kl_term() + self.fc_3.kl_term())
        loss = self.alpha * log_det_i + nll_i + kl_i
        # print(self.alpha*log_det_i)
        # print(nll_i)
        # print(self.beta*kl_i)
        self.test_acc_epoch(mu, y)
        self.log_dict({'train_acc': self.train_acc_epoch.compute(), 'test_acc': self.test_acc_epoch.compute()}, prog_bar=True, on_step=False, on_epoch=True)

    #def training_epoch_end(self, outs):
     #   self.train_acc_epoch.reset()

    def validation_epoch_end(self, outs):
        self.train_acc_epoch.reset()
        self.test_acc_epoch.reset()

        # if self.epoch_counter > 1:
        #     mu_1 = pd.DataFrame(self.conv_1.mu.weight.view(-1, 1).detach().cpu().numpy(), columns=['conv_1'])
        #     mu_2 = pd.DataFrame(self.conv_2.mu.weight.view(-1, 1).detach().cpu().numpy(), columns=['conv_2'])
        #     mu_3 = pd.DataFrame(self.linear_1.mu.weight.view(-1, 1).detach().cpu().numpy(), columns=['linear_1'])
        #     mu_4 = pd.DataFrame(self.linear_2.mu.weight.view(-1, 1).detach().cpu().numpy(), columns=['linear_2'])
        #     sns.kdeplot(mu_1['conv_1'], shade=True, label='conv_1')
        #     sns.kdeplot(mu_2['conv_2'], shade=True, label='conv_2')
        #     sns.kdeplot(mu_3['linear_1'], shade=True, label='linear_1')
        #     sns.kdeplot(mu_4['linear_2'], shade=True, label='linear_2')
        #     plt.legend()
        #     plt.title('Mu Epoch {}'.format(self.epoch_counter))
        #     plt.savefig('cifar/mu_epoch_{}.png'.format(self.epoch_counter))
        #     plt.show()
        #     plt.clf()
        #     sig_1 = pd.DataFrame(self.conv_1.sigma.weight.view(-1, 1).detach().cpu().numpy(), columns=['conv_1'])
        #     sig_2 = pd.DataFrame(self.conv_2.sigma.weight.view(-1, 1).detach().cpu().numpy(), columns=['conv_2'])
        #     sig_3 = pd.DataFrame(self.linear_1.sigma.weight.view(-1, 1).detach().cpu().numpy(), columns=['linear_1'])
        #     sig_4 = pd.DataFrame(self.linear_2.sigma.weight.view(-1, 1).detach().cpu().numpy(), columns=['linear_2'])
        #     sns.kdeplot(sig_1['conv_1'], shade=True, label='conv_1')
        #     sns.kdeplot(sig_2['conv_2'], shade=True, label='conv_2')
        #     sns.kdeplot(sig_3['linear_1'], shade=True, label='linear_1')
        #     sns.kdeplot(sig_4['linear_2'], shade=True, label='linear_2')
        #     plt.legend()
        #     plt.title('Sigma Epoch {}'.format(self.epoch_counter))
        #     plt.savefig('cifar/sigma_epoch_{}.png'.format(self.epoch_counter))
        #     plt.show()
        #     plt.clf()
        # self.epoch_counter += 1

    def train_dataloader(self):
        # transforms
        # prepare transforms standard to MNIST
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        # data
        mnist_train = CIFAR10(os.getcwd(), train=True, download=True, transform=transform)
        return DataLoader(mnist_train, batch_size=self.batch_size, num_workers=2, shuffle=True, pin_memory=True)

    def val_dataloader(self):
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        mnist_test = CIFAR10(os.getcwd(), train=False, download=True, transform=transform)
        return DataLoader(mnist_test, batch_size=self.batch_size, num_workers=2, shuffle=False, pin_memory=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3, weight_decay=0.001, amsgrad=True)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=1)
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'train_loss'
        }

    def get_progress_bar_dict(self):
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items


def main():
    no_epochs = 25
    vanilla_batch = 400   # This is the entire size of the training set.
    model = ConvNet(batch_size=vanilla_batch)   # Default batch size will get overwritten by the auto_scale_batch_size method
    trainer = pl.Trainer(gpus=1, max_epochs=no_epochs, auto_scale_batch_size=None, check_val_every_n_epoch=1)   # Auto scale arguments can be: [None, 'power', 'binsearch']
    trainer.tune(model)    # This does the auto scaling
    trainer.fit(model)    # This does the training and validation
    # trainer.test(model)    # Runs the test data (if you had one)
    torch.save(model.state_dict(), 'vdp_conv_cifar10.pt')
    pass


if __name__ == '__main__':
    main()
    # To see the things you logged in tensorboard run the following command in the directory of the file
    # tensorboard --logdir=./lightning_logs --port=6006