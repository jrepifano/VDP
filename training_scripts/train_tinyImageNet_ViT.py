import vdp
import torch
import wandb
import argparse
import torchvision.transforms
import pytorch_lightning as pl
from torchmetrics import Accuracy
from torch.utils.data import DataLoader
from simple_vit_vdp import SimpleViT_vdp
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from torchvision.transforms import AutoAugment
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--lr', help='Learning Rate', type=float, default=5e-2, required=False)
    parser.add_argument(
        '--wd', help='Weight Decay', type=float, default=0.0005, required=False)
    parser.add_argument(
        '--optim', help='Optimizer Type (adam, sgd)', type=str, default="adam")
    parser.add_argument(
        '--sched', help='Scheduler Type (multisteplr, cosine, plateau, onecycle)', type=str, default="reduceLROnPlateau")
    parser.add_argument(
        '--batch_size', help='Batch Size', type=int, default=512)
    parser.add_argument(
        '--alpha', help="Hyperparameter for scaling nll loss term", type=int, default=1)
    parser.add_argument(
        '--tau', help="Hyperparameter for tuning KL loss term", type=float, default=0.01)
    args = parser.parse_args()
    return args


class ViT_vdp(SimpleViT_vdp, pl.LightningModule):
    def __init__(self, config, num_classes=200):
        super().__init__(image_size = 256, patch_size = 32, channels = 3, num_classes = num_classes, dim = 1024, depth = 6, heads = 16, mlp_dim = 2048)
        self.lr = config.lr
        self.wd = config.wd
        self.alpha = config.alpha
        self.tau = config.tau
        # self.batch_size = config.batch_size
        self.batch_size = 64
        self.optim = config.optim
        self.sched = config.sched

        self.num_classes= num_classes
        self.train_acc = Accuracy(task='multiclass', num_classes=num_classes)
        self.val_acc = Accuracy(task='multiclass', num_classes=num_classes)
        self.test_acc = Accuracy(task='multiclass', num_classes=num_classes)

    def get_loss(self, mu, sigma, y):
        log_det, nll = vdp.ELBOLoss(mu, sigma, y, self.num_classes)
        kl = vdp.gather_kl(self)
        wandb.log({"log_det": log_det,
                  "nll": self.alpha * nll,
                  "kl": self.tau * sum(kl)})
        loss = log_det + self.alpha * nll + self.tau * sum(kl)
        # loss = log_det + 1000 * nll + sum(kl)
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
            AutoAugment(policy=torchvision.transforms.AutoAugmentPolicy.IMAGENET),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        train = ImageFolder('/data/tiny-imagenet-200/train', transform=transform)
        return DataLoader(train, batch_size=self.batch_size, num_workers=8, shuffle=True, pin_memory=True)


    def val_dataloader(self):
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        test = ImageFolder('/data/tiny-imagenet-200/val', transform=transform)
        return DataLoader(test, batch_size=self.batch_size, num_workers=8, shuffle=False, pin_memory=True)

    def test_dataloader(self):
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        test = ImageFolder('/data/tiny-imagenet-200/test', transform=transform)
        return DataLoader(test, batch_size=self.batch_size, num_workers=8, shuffle=False, pin_memory=True)

    def configure_optimizers(self):
        if self.optim == "adam":
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.wd, amsgrad=True)
        else:
            optimizer = torch.optim.SGD(self.parameters(), lr=self.lr,
                                        momentum=0.9, weight_decay=self.wd)

        if self.sched == "multiStep":
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [40, 80, 120])
        elif self.sched == "reduceLROnPlateau":            
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2)
        elif self.sched == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'loss'
        }


def train(model, epochs):

    wandb_logger = WandbLogger()
    early_stop_callback = EarlyStopping(monitor="loss", min_delta=0.00, patience=5, verbose=False, mode="min")
    checkpoint_callback = ModelCheckpoint(monitor="val_acc", mode="max")
    trainer = pl.Trainer(gpus=[0], max_epochs=epochs, check_val_every_n_epoch=5, logger=wandb_logger, #auto_scale_batch_size='power',
                         accelerator=None, callbacks=[early_stop_callback, checkpoint_callback], inference_mode=False)
    trainer.tune(model)
    trainer.fit(model)
    out = trainer.test(model)
    test_acc = out[0]['test_acc_epoch']
    # torch.save(model.state_dict(), save_dir)
    # print("Saving " + save_dir)

if __name__ =='__main__':
    cmd =parse_args()
    wandb.init(config=cmd, project="ViT_VDP_Tiny_ImageNet")
    config = wandb.config

    model = ViT_vdp(config) # 512 local
    train(model, 209)
