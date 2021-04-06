import os
import vdp
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification, load_iris
from sklearn.model_selection import train_test_split


os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # mnist images are (1, 28, 28) (channels, width, height)
        self.layer_1 = vdp.Linear(4, 50, input_flag=True)
        self.layer_2 = vdp.Linear(50, 3)
        self.relu = vdp.ReLU()
        self.softmax = vdp.Softmax()

    def forward(self, x):
        mu, sigma = self.layer_1(x)
        mu, sigma = self.relu(mu, sigma)
        mu, sigma = self.layer_2(mu, sigma)
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
    X, y = load_iris(return_X_y=True)
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = Model()
    no_epochs = 2000
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.to('cuda:0')
    train_accs = list()
    test_accs = list()
    train_loss = list()
    test_loss = list()
    logdet = list()
    likelihood = list()
    kl = list()
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    x_train, y_train = torch.from_numpy(x_train).float().to('cuda:0'), torch.from_numpy(y_train).long().to('cuda:0')
    x_test, y_test = torch.from_numpy(x_test).float().to('cuda:0'), torch.from_numpy(y_test).long().to('cuda:0')
    model.train()
    print_iter = 100
    for epoch in range(no_epochs):
        optimizer.zero_grad()
        mu, sigma = model.forward(x_train)
        log_det_i, likelihood_i = vdp.ELBOLoss(mu, sigma, y_train.to('cuda:0'))
        kl_i = 0.001*(model.layer_1.kl_term()+model.layer_2.kl_term())
        loss = 0.01*log_det_i+likelihood_i-kl_i
        train_loss.append(loss.item())
        loss.backward()
        optimizer.step()
        train_acc = model.score(mu, y_train)
        train_accs.append(train_acc)
        model.eval()  # This removes stuff like dropout and batch norm for inference stuff
        mu, sigma = model.forward(x_test)
        log_det_i, likelihood_i = vdp.ELBOLoss(mu, sigma, y_test .to('cuda:0'))
        loss = 0.1*log_det_i+likelihood_i-kl_i
        test_loss.append(loss.item())
        test_acc = model.score(mu, y_test)
        test_accs.append(test_acc)
        logdet.append(log_det_i.detach().cpu().numpy())
        likelihood.append(likelihood_i.detach().cpu().numpy())
        kl.append(kl_i.detach().cpu().numpy())
        if epoch % print_iter == 0 or epoch == no_epochs-1:
            print('Epoch {}/{}: Training Loss: {:.2f}'.format(epoch + 1, no_epochs, loss.item()))
            print('Train Accuracy: {:.2f}'.format(train_acc))
            print('Test Accuracy: {:.2f}'.format(test_acc))
    plt.plot(np.arange(no_epochs), train_accs, label='Train Accuracy')
    plt.plot(np.arange(no_epochs), test_accs, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Small Network VDP++')
    plt.legend()
    plt.show()
    plt.plot(np.arange(no_epochs), train_loss, label='Train Loss')
    plt.plot(np.arange(no_epochs), test_loss, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('loss')
    plt.title('Small Network VDP++')
    plt.legend()
    plt.show()

    plt.plot(np.arange(no_epochs), logdet, label='Log determinant')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Fashion MNIST: Small Network VDP++')
    plt.legend()
    plt.show()
    plt.plot(np.arange(no_epochs), likelihood, label='Likelihood')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Fashion MNIST: Small Network VDP++')
    plt.legend()
    plt.show()
    plt.plot(np.arange(no_epochs), kl, label='KL terms')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Fashion MNIST: Small Network VDP++')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
