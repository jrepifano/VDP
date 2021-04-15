import torch


def softplus(a):
    return torch.log(1.+torch.exp(torch.clamp(a, min=-88, max=88)))


class Linear(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=True, input_flag=False):
        super(Linear, self).__init__()
        self.input_flag = input_flag
        self.mu = torch.nn.Linear(in_features, out_features, bias)
        self.sigma = torch.nn.Linear(in_features, out_features, bias)
        torch.nn.init.xavier_normal_(self.mu.weight)
        torch.nn.init.uniform_(self.sigma.weight, a=0, b=5)


    def forward(self, mu_x, sigma_x=torch.tensor(0., requires_grad=True)):
        if self.input_flag:
            mu_y = self.mu(mu_x)
            sigma_y = mu_x ** 2 @ softplus(self.sigma.weight).T + self.sigma.bias
            pass
        else:
            mu_y = self.mu(mu_x)
            sigma_y = (softplus(self.sigma.weight) @ sigma_x.T).T + \
                      (self.mu.weight**2 @ sigma_x.T).T + \
                      (mu_x ** 2 @ softplus(self.sigma.weight).T) + self.sigma.bias
            pass
        return mu_y, sigma_y

    def kl_term(self):
        kl = 0.5*torch.mean(self.mu.weight.shape[1] * softplus(self.sigma.weight) + torch.norm(self.mu.weight)**2
                            - self.mu.weight.shape[1] - self.mu.weight.shape[1] * torch.log(softplus(self.sigma.weight)))
        return kl


class ReLU(torch.nn.Module):
    def __init__(self):
        super(ReLU, self).__init__()
        self.relu = torch.nn.ReLU()

    def forward(self, mu, sigma):
        mu_a = self.relu(mu)
        sigma_a = sigma * (torch.autograd.grad(torch.sum(mu_a), mu, create_graph=True, retain_graph=True)[0]**2)
        return mu_a, sigma_a


class Softmax(torch.nn.Module):
    def __init__(self):
        super(Softmax, self).__init__()
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, mu, sigma):
        mu = self.softmax(mu)
        # print(mu)
        J = mu*(1-mu)
        sigma = (J**2) * sigma
        return mu, sigma


def ELBOLoss(mu, sigma, y):
    y_hot = torch.nn.functional.one_hot(y, num_classes=10)
    num_samples = y_hot.shape[0]
    sigma_clamped = torch.log(1+torch.exp(torch.clamp(sigma, 0, 88)))
    # print(torch.sum(sigma))
    # print(torch.sum(sigma_clamped))
    log_det = torch.mean(torch.log(torch.prod(sigma_clamped, dim=1)))
    log_likelihood = torch.mean(((y_hot-mu)**2).T @ torch.reciprocal(sigma_clamped))
    return log_det, log_likelihood
