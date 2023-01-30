import os
import sys
sys.path.append(os.getcwd())
import vdp
import torch
import resnet_vdp
import torchvision.transforms
import pytorch_lightning as pl
from torchmetrics import Accuracy
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import transforms
from torchvision.transforms import AutoAugment
from pytorch_lightning.callbacks.early_stopping import EarlyStopping


class sto_cifar10(resnet_vdp.ResNet):
    def __init__(self, lr, wd, pt, batch_size=1000):
        super().__init__(block=resnet_vdp.block, layers=[2, 2, 2, 2], image_channels=3, num_classes=10)
        self.lr = lr
        self.wd = wd
        self.pt = pt
        self.batch_size = batch_size
        self.train_acc = Accuracy(task='multiclass', num_classes=10)
        self.val_acc = Accuracy(task='multiclass', num_classes=10)
        self.test_acc = Accuracy(task='multiclass', num_classes=10)
        self.alpha, self.tau = None, None

    def get_loss(self, mu, sigma, y):
        log_det, nll = vdp.ELBOLoss(mu, sigma, y)
        kl = vdp.gather_kl(self)
        # if self.alpha is None:
        #     self.alpha, self.tau = vdp.scale_hyperp(log_det, nll, kl)
        # loss = self.alpha * log_det + nll + self.tau * sum(kl)
        loss = log_det + 4395 * nll + 75.703 * sum(kl)
        return loss

    @torch.enable_grad()
    def training_step(self, batch, batch_idx):
        x, y = batch
        mu, sigma = self(x)
        loss = self.get_loss(mu, sigma, y)
        self.log('loss', loss)
        self.train_acc(mu.softmax(dim=-1), y)
        self.log('train_acc', self.train_acc, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    @torch.enable_grad()
    def on_train_epoch_end(self):
        self.log('train_acc', self.train_acc)

    @torch.enable_grad()
    def validation_step(self, batch, batch_idx):
        x, y = batch
        mu, sigma = self(x)
        loss = self.get_loss(mu, sigma, y)
        self.log('val_loss', loss)
        self.val_acc(mu.softmax(dim=-1), y)
        self.log('val_acc', self.val_acc, on_step=True, on_epoch=True, prog_bar=True)

    @torch.enable_grad()
    def test_step(self, batch, batch_idx):
        x, y = batch
        mu, sigma = self(x)
        loss = self.get_loss(mu, sigma, y)
        self.log('test_loss', loss)
        self.test_acc(mu.softmax(dim=-1), y)
        self.log('test_acc', self.test_acc, on_step=True, on_epoch=True)

    @torch.enable_grad()
    def on_test_epoch_end(self):
        self.log('test_acc', self.test_acc)

    def train_dataloader(self):
        transform = transforms.Compose([
            #RandAugment(),
            AutoAugment(policy=torchvision.transforms.AutoAugmentPolicy.CIFAR10),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        train = CIFAR10(os.getcwd(), train=True, download=True, transform=transform)
        return DataLoader(train, batch_size=self.batch_size, num_workers=8, shuffle=True, pin_memory=True)

    def val_dataloader(self):
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        test = CIFAR10(os.getcwd(), train=False, download=True, transform=transform)
        return DataLoader(test, batch_size=self.batch_size, num_workers=8, shuffle=False, pin_memory=True)

    def test_dataloader(self):
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        test = CIFAR10(os.getcwd(), train=False, download=True, transform=transform)
        return DataLoader(test, batch_size=self.batch_size, num_workers=8, shuffle=False, pin_memory=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.wd, amsgrad=True)
        # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [40, 80, 120])
        # scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
        # optimizer = torch.optim.SGD(self.parameters(), lr=self.lr,
        #               momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'loss'
        }


def train(model, epochs):
    # wandb_logger = WandbLogger()
    early_stop_callback = EarlyStopping(monitor="loss", min_delta=0.00, patience=20, verbose=False, mode="min")
    trainer = pl.Trainer(gpus=[0], max_epochs=epochs, check_val_every_n_epoch=5,  #auto_scale_batch_size='power',
                         accelerator=None, callbacks=[early_stop_callback], inference_mode=False)
    trainer.tune(model)
    trainer.fit(model)
    out = trainer.test(model)
    test_acc = out[0]['test_acc_epoch']
    # torch.save(model.state_dict(), save_dir)
    # print("Saving " + save_dir)



if __name__ =='__main__':

    model = sto_cifar10(0.0001677, 0.001266, 10, 128) # 512 local
    train(model, 209)
