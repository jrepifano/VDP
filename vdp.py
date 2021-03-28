import torch


def logexp(a):
    return torch.log(1.+torch.exp(torch.clamp(a, min=-88, max=88)))


class Linear(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=True, input_flag=False):
        super(Linear, self).__init__()
        self.input_flag = input_flag
        self.mu = torch.nn.Linear(in_features, out_features, bias)
        self.sigma = torch.nn.Linear(in_features, out_features, bias)
        torch.nn.init.normal_(self.mu.weight, mean=0, std=0.1)
        torch.nn.init.uniform_(self.sigma.weight, a=-12, b=-2.2)

    def forward(self, mu_x, sigma_x=torch.tensor(0)):
        if self.input_flag:
            mu_y = self.mu(mu_x)
            sigma_y = (mu_x**2 @ logexp(self.sigma.weight).T) + self.sigma.bias
            pass
        else:
            mu_y = self.mu(mu_x)
            sigma_y = (logexp(sigma_x) @ logexp(self.sigma.weight).T) +\
                      (mu_x**2 @ logexp(self.sigma.weight).T) + \
                      (self.mu.weight**2 @ logexp(sigma_x).T).T + self.sigma.bias
            pass
        return mu_y, sigma_y

    def kl_term(self):
        kl = 0.5*torch.mean(self.mu.weight.shape[0]*logexp(self.sigma.weight)+torch.norm(self.mu.weight)**2
                           - self.mu.weight.shape[0]-self.mu.weight.shape[0]*logexp(self.sigma.weight))
        return kl


class ReLU(torch.nn.Module):
    def __init__(self):
        super(ReLU, self).__init__()
        self.relu = torch.nn.ReLU()

    def forward(self, mu, sigma):
        mu = self.relu(mu)
        sigma = sigma * torch.autograd.grad(torch.sum(mu), mu)[0]**2
        return mu, sigma


class Softmax(torch.nn.Module):
    def __init__(self):
        super(Softmax, self).__init__()
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, mu, sigma):
        # This is sorta incorrect. It will sum the rows of the product instead of
        # giving us just the diagonal elements of the product...
        # need to find a way to compute J**2 @ sigma without directly computing J
        jvp = torch.autograd.functional.jvp(self.softmax, mu, sigma)
        sigma = torch.autograd.functional.vjp(self.softmax, mu, jvp[1])[1]
        mu = self.softmax(mu)
        return mu, sigma


def ELBOLoss(mu, sigma, y):
    N = len(mu)
    l = len(torch.unique(y))
    # y_hot = torch.nn.functional.one_hot(y)
    pi = (torch.acos(torch.zeros(1))*2).to('cuda:0')    # which is 3.1415927410125732
    # -((N * l) / 2) * torch.log(2 * pi) * ((N / 2) *
    criterion = torch.nn.CrossEntropyLoss(reduction='none')
    sigma_clamped = torch.clamp(sigma, 1e-10, 1e10)
    constant = -((N * l) / 2) * (torch.log(2 * pi))    # Constant breaks network
    log_det = (N / 2) * torch.log(1e-3+torch.prod(sigma_clamped, dim=1))
    likelihood = 0.5*torch.sum(((criterion(mu, y).unsqueeze(1).repeat_interleave(len(torch.unique(y)), dim=1)**2).unsqueeze(1) @ torch.reciprocal(sigma_clamped).unsqueeze(2)).view(-1))
    loss = (log_det+likelihood)
    return torch.mean(loss)
