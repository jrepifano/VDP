import os
import vdp
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'  # Set this flag to set your devices. For example if I set '6,7', then cuda:0 and cuda:1 in code will be cuda:6 and cuda:7 on hardware


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # mnist images are (1, 28, 28) (channels, width, height)
        self.layer_1 = vdp.Linear(50, 128, input_flag=True)
        self.layer_2 = vdp.Linear(128, 2)
        self.bn1 = torch.nn.BatchNorm1d(128)
        self.bn2 = torch.nn.BatchNorm1d(2)
        self.relu = vdp.ReLU()
        self.softmax = vdp.Softmax()

    def forward(self, x):
        mu, sigma = self.layer_1(x)
        print(sigma)
        mu = self.bn1(mu)
        mu, sigma = self.relu(mu, sigma)
        print(sigma)
        mu, sigma = self.layer_2(mu, sigma)
        print(sigma)
        mu = self.bn2(mu)
        mu, sigma = self.softmax(mu, sigma)
        print(sigma)
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


def gen_data():
    # n_samples = np.random.randint(100, 5000)
    n_samples = 10000
    print('Number of Samples in DS: ' + str(n_samples))
    # n_feats = np.random.choice([10, 20, 50, 100], 1).item()
    n_feats = 50
    n_clusters = np.random.randint(2, 14)
    sep = 5 * np.random.random_sample()
    hyper = np.random.choice([True, False], 1).item()
    X, y = make_classification(n_samples, n_features=n_feats, n_informative=n_feats,
                               n_redundant=0, n_repeated=0, n_classes=2, n_clusters_per_class=n_clusters,
                               weights=None, flip_y=0, class_sep=sep, hypercube=hyper, shift=0, scale=1, shuffle=False)
    X, x_test, y, y_test = train_test_split(X, y, test_size=0.2)
    return X, x_test, y, y_test


def main():
    x_train, x_test, y_train, y_test = gen_data()
    model = Model()
    no_epochs = 20
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    model.to('cuda:0')
    train_accs = []
    test_accs = []
    x_train, y_train = torch.from_numpy(x_train).float().to('cuda:0'), torch.from_numpy(y_train).long().to('cuda:0')
    x_test, y_test = torch.from_numpy(x_test).float().to('cuda:0'), torch.from_numpy(y_test).long().to('cuda:0')
    for epoch in range(no_epochs):
        model.train()
        optimizer.zero_grad()
        mu, sigma = model.forward(x_train)
        loss = vdp.ELBOLoss(mu, sigma, y_train)+0.0002*(model.layer_1.kl_term()+model.layer_2.kl_term())
        loss.backward()
        optimizer.step()
        train_acc = model.score(mu, y_train)
        print('Epoch {}/{}: Training Loss: {:.2f}'.format(epoch + 1, no_epochs, loss.item()))
        print('Train Accuracy: {:.2f}'.format(train_acc))
        train_accs.append(train_acc)
        model.eval()  # This removes stuff like dropout and batch norm for inference stuff
        mu, sigma = model.forward(x_test)
        loss = vdp.ELBOLoss(mu, sigma, y_test)+0.0002*(model.layer_1.kl_term()+model.layer_2.kl_term())
        test_acc = model.score(mu, y_test)
        print('Test Loss: {:.2f},    Test Accuracy: {:.2f}'.format(loss.item(), test_acc))
        test_accs.append(test_acc)
    plt.plot(np.arange(no_epochs), train_accs, label='Train Accuracy')
    plt.plot(np.arange(no_epochs), test_accs, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Small Network VDP++')
    plt.legend()
    plt.show()
    snrs = [-6, -3, 1, 5, 10, 20]
    sigmas = list()
    model.eval()
    for snr in range(len(snrs)):
        print('SNR: {}'.format(snrs[snr]))
        x_noisy = add_noise(x_test.cpu().numpy(), snrs[snr])
        x_noisy = torch.from_numpy(x_noisy).float().to('cuda:0')
        mu, sigma = model.forward(x_noisy)
        sigmas.append(np.mean(np.mean(np.abs(sigma.detach().cpu().numpy()), axis=1)))
    plt.figure()
    plt.plot(snrs, sigmas)
    plt.xlabel('SNR (dB)')
    plt.ylabel('Test Sigma')
    plt.title('Small network VDP++')
    plt.show()
    pass


if __name__ == '__main__':
    main()
