import torch
from torch.autograd.functional import jacobian, jvp, vjp


def square_sum(x, y):
    return x**2+y**2

#   3 Training Instances
x = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [1, 2, 3, 4, 5]])
#   2 Sigmas
sigma_combined = torch.tensor([[6, 7, 8, 9, 10], [1, 2, 3, 4, 5]])
sigma_z = x**2 @ sigma_combined.T
print(sigma_z)
for i in range(len(x)):
    x_i = x[i]
    for j in range(len(sigma_combined)):
        sigma_j = torch.diag(sigma_combined[j])
        sigma_ij = x_i.T @ sigma_j @ x_i
        print(sigma_ij)


v = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]).float()
J = jacobian(square_sum, (v, v))[1]
print(J)
jvp_ = jvp(square_sum, (v, v), (v, v))[1]
print(jvp_)
vjp_ = vjp(square_sum, (v, v), v)[1]
print(vjp_)
