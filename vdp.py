import math
import torch


def softplus(a):
    return torch.log(1.+torch.exp(torch.clamp(a, min=-88, max=88)))


def retrieve_elements_from_indices(tensor, indices):
    flattened_tensor = tensor.flatten(start_dim=2)
    output = flattened_tensor.gather(dim=2, index=indices.flatten(start_dim=2)).view_as(indices)
    return output


class Linear(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=True, input_flag=False):
        super(Linear, self).__init__()
        self.input_flag = input_flag
        self.bias = bias
        self.mu = torch.nn.Linear(in_features, out_features, bias)
        self.sigma = torch.nn.Linear(in_features, out_features, bias)
        torch.nn.init.xavier_normal_(self.mu.weight)
        torch.nn.init.uniform_(self.sigma.weight, a=0, b=5)

    def forward(self, mu_x, sigma_x=torch.tensor(0., requires_grad=True)):
        if self.input_flag:
            mu_y = self.mu(mu_x)
            sigma_y = mu_x ** 2 @ softplus(self.sigma.weight).T
            pass
        else:
            mu_y = self.mu(mu_x)
            if len(sigma_x.shape) > 2:
                sigma_y = (softplus(self.sigma.weight) @ sigma_x.mT).mT + \
                          (self.mu.weight**2 @ sigma_x.mT).mT + \
                          (mu_x ** 2 @ softplus(self.sigma.weight).mT)
            else:
                sigma_y = (softplus(self.sigma.weight) @ sigma_x.T).T + \
                        (self.mu.weight**2 @ sigma_x.T).T + \
                        (mu_x ** 2 @ softplus(self.sigma.weight).T)
            pass
        if self.bias:
            sigma_y += self.sigma.bias
        sigma_y *= 1e-3
        return mu_y, sigma_y

    def kl_term(self):
        kl = 0.5*torch.mean(self.mu.weight.shape[1] * softplus(self.sigma.weight) + torch.norm(self.mu.weight)**2
                            - self.mu.weight.shape[1] - self.mu.weight.shape[1] * torch.log(softplus(self.sigma.weight)))
        return kl


class Conv2d(torch.nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size=(5, 5), stride=1, padding=0, dilation=1,
                 groups=1, bias=True, padding_mode='zeros',
                 input_flag=False):
        super(Conv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.input_flag = input_flag
        self.mu = torch.nn.Conv2d(in_channels, out_channels,
                                  kernel_size, stride, padding, dilation,
                                  groups, bias, padding_mode)
        torch.nn.init.xavier_normal_(self.mu.weight)
        self.sigma = torch.nn.Linear(1, out_channels, bias=bias)
        torch.nn.init.uniform_(self.sigma.weight, a=0, b=5)

        self.unfold = torch.nn.Unfold(kernel_size, dilation, padding, stride)

    def forward(self, mu_x, sigma_x=torch.tensor(0., requires_grad=True)):
        if self.input_flag:
            mu_y = self.mu(mu_x)
            vec_x = self.unfold(mu_x)
            sigma_y = softplus(self.sigma.weight).repeat(1, vec_x.shape[1]) @ vec_x ** 2
            sigma_y = sigma_y.view(mu_y.shape[0], mu_y.shape[1], mu_y.shape[2], mu_y.shape[3])
            pass
        else:
            mu_y = self.mu(mu_x)
            vec_x = self.unfold(mu_x)
            sigma_y = (softplus(self.sigma.weight).repeat(1, vec_x.shape[1]) @ self.unfold(sigma_x)) + \
                      (self.mu.weight.view(self.out_channels, -1)**2 @ self.unfold(sigma_x).permute(2, 1, 0)).permute(2, 1, 0) + \
                      (softplus(self.sigma.weight).repeat(1, vec_x.shape[1]) @ (vec_x ** 2).permute(2, 1, 0)).permute(2, 1, 0)
            sigma_y = sigma_y.view(mu_y.shape[0], mu_y.shape[1], mu_y.shape[2], mu_y.shape[3])
        sigma_y *= 1e-3
        return mu_y, sigma_y

    def kl_term(self):
        kl = 0.5*torch.mean(self.kernel_size * softplus(self.sigma.weight) + torch.norm(self.mu.weight)**2
                            - self.kernel_size - self.kernel_size * torch.log(softplus(self.sigma.weight)))
        return kl


class MaxPool2d(torch.nn.Module):
    def __init__(self, kernel_size=2, stride=2, padding=0, dilation=1,
                 return_indices=True, ceil_mode=False):
        super(MaxPool2d, self).__init__()
        self.mu = torch.nn.MaxPool2d(kernel_size, stride, padding, dilation, return_indices, ceil_mode)

    def forward(self, mu_x, sigma_x):
        mu_y, where = self.mu(mu_x)
        sigma_y = retrieve_elements_from_indices(sigma_x, where)
        return mu_y, sigma_y


class ReLU(torch.nn.Module):
    def __init__(self):
        super(ReLU, self).__init__()
        self.relu = torch.nn.SELU()

    def forward(self, mu, sigma):
        mu_a = self.relu(mu)
        sigma_a = sigma * (torch.autograd.grad(torch.sum(mu_a), mu, create_graph=True, retain_graph=True)[0]**2)
        return mu_a, sigma_a
    
class GELU(torch.nn.Module):
    def __init__(self):
        super(GELU, self).__init__()
        self.gelu = torch.nn.GELU()

    def forward(self, mu, sigma):
        mu_a = self.gelu(mu)
        sigma_a = sigma * (torch.autograd.grad(torch.sum(mu_a), mu, create_graph=True, retain_graph=True)[0]**2)
        return mu_a, sigma_a
    
class Sigmoid(torch.nn.Module):
    def __init__(self):
        super(Sigmoid, self).__init__()
        self.sigmoid = torch.nn.Sigmoid()
    
    def forward(self, mu, sigma):
        mu_a = self.sigmoid(mu)
        sigma_a = sigma * (torch.autograd.grad(torch.sum(mu_a), mu, create_graph=True, retain_graph=True)[0]**2)
        return mu_a, sigma_a


class Softmax(torch.nn.Module):
    def __init__(self, dim=1):
        super(Softmax, self).__init__()
        self.softmax = torch.nn.Softmax(dim=dim)

    def forward(self, mu, sigma):
        mu = self.softmax(mu)
        J = mu*(1-mu)
        sigma = (J**2) * sigma
        return mu, sigma


class BatchNorm2d(torch.nn.Module):
    def __init__(self, num_features):
        super(BatchNorm2d, self).__init__()
        self.mu = torch.nn.BatchNorm2d(num_features)

    def forward(self, mu, sigma):
        mu_a = self.mu(mu)
        bn_param = ((self.mu.weight/(torch.sqrt(self.mu.running_var + self.mu.eps))) ** 2)
        sigma_a = bn_param[None, :, None, None] * sigma
        return mu_a, sigma_a
    

class LayerNorm(torch.nn.Module):
    def __init__(self, normalized_shape):
        super(LayerNorm, self).__init__()
        self.mu = torch.nn.LayerNorm(normalized_shape)

    def forward(self, mu, sigma):
        mu_a = self.mu(mu)
        bn_param = ((self.mu.weight.unsqueeze(0)/(torch.sqrt(mu.var((-1), keepdim=True, unbiased=False) + self.mu.eps))) ** 2)
        sigma_a = bn_param * sigma
        return mu_a, sigma_a
    

def ELBOLoss(mu, sigma, y, num_classes=10, criterion='nll'):
    y_hot = torch.nn.functional.one_hot(y, num_classes=num_classes)
    sigma_clamped = torch.log(1+torch.exp(torch.clamp(sigma, 0, 88)))
    if num_classes > 10:
        log_det = torch.mean(torch.log(torch.log(torch.sum(sigma_clamped, dim=1))))
    else:   
        log_det = torch.mean(torch.log(torch.prod(sigma_clamped, dim=1)))
    if criterion == 'nll':
        nll = torch.mean(((y_hot-mu)**2).T @ torch.reciprocal(sigma_clamped))
    elif criterion == 'ce':
        criterion = torch.nn.CrossEntropyLoss()
        nll = criterion(mu, y)
    return log_det, nll


def orderOfMagnitude(number):
    return math.floor(math.log(number, 10))


def scale_hyperp(log_det, nll, kl):
    # Find the alpha scaling factor
    lli_power = orderOfMagnitude(nll)
    ldi_power = orderOfMagnitude(log_det)
    alpha = 10**(lli_power - ldi_power)     # log_det_i needs to be 1 less power than log_likelihood_i

    beta = list()
    # Find scaling factor for each kl term
    kl = [i.item() for i in kl]
    smallest_power = orderOfMagnitude(min(kl))
    for i in range(len(kl)):
        power = orderOfMagnitude(kl[i])
        power = smallest_power-power
        beta.append(1)
        # beta.append(10.0**power)

    # Find the tau scaling factor
    tau = 10**(smallest_power - lli_power)

    return alpha, tau


def gather_kl(model):
    kl = list()
    for layer in model.modules():
        if hasattr(layer, 'kl_term'):
            kl.append(layer.kl_term())
    return kl


class AdaptiveAvgPool2d(torch.nn.Module):
    def init(self, shape=(1,1)):
        super(AdaptiveAvgPool2d, self).init()
        self.avgpool = torch.nn.AdaptiveAvgPool2d(shape)

    def forward(self, mu, sigma):
        mu_y = self.avgpool(mu)
        sigma_y = self.avgpool(sigma) + torch.var(mu)
        return mu_y, sigma_y