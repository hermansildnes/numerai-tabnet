import torch
import torch.nn as nn
import torch.nn.functional as F


class GBN(nn.Module):
    def __init__(self, inp, vbs=128, momentum=0.01):
        super().__init__()
        self.bn = nn.BatchNorm1d(inp, momentum=momentum)
        self.vbs = vbs

    def forward(self, x):
        if x.size(0) <= self.vbs:
            return self.bn(x)
        chunk = torch.chunk(x, x.size(0) // self.vbs, 0)
        res = [self.bn(y) for y in chunk]
        return torch.cat(res, 0)


class Sparsemax(nn.Module):
    """Sparsemax activation function."""

    def __init__(self, dim=-1):
        super(Sparsemax, self).__init__()
        self.dim = dim

    def forward(self, input):
        """Forward function."""
        original_size = input.size()
        input = input.view(-1, input.size(self.dim))

        dim = 1
        number_of_logits = input.size(dim)

        # Translate input by max for numerical stability
        input = input - torch.max(input, dim=dim, keepdim=True)[0].expand_as(input)

        # Sort input in descending order.
        zs = torch.sort(input=input, dim=dim, descending=True)[0]
        range_tensor = torch.arange(
            start=1,
            end=number_of_logits + 1,
            step=1,
            dtype=input.dtype,
            device=input.device,
        )
        range_tensor = range_tensor.view(1, -1)
        range_tensor = range_tensor.expand_as(zs)

        # Determine sparsity of projection
        bound = 1 + range_tensor * zs
        cumulative_sum_zs = torch.cumsum(zs, dim)
        is_gt = torch.gt(bound, cumulative_sum_zs).type(input.type())
        k = torch.max(is_gt * range_tensor, dim, keepdim=True)[0]

        # Compute threshold function
        zs_sparse = is_gt * zs

        # Compute taus
        taus = (torch.sum(zs_sparse, dim, keepdim=True) - 1) / k
        taus = taus.expand_as(input)

        # Sparsemax
        self.output = torch.max(torch.zeros_like(input), input - taus)

        # Reshape back to original shape
        output = self.output.view(original_size)

        return output


class AttentionTransformer(nn.Module):
    def __init__(self, d_a, inp_dim, relax, vbs=128):
        super().__init__()
        self.fc = nn.Linear(d_a, inp_dim)
        self.bn = GBN(inp_dim, vbs=vbs)
        self.smax = Sparsemax(dim=-1)
        self.r = relax

    def forward(self, a, priors):
        a = self.bn(self.fc(a))
        mask = self.smax(a * priors)
        priors = priors * (self.r - mask)
        return mask, priors


class GLU(nn.Module):
    def __init__(self, inp_dim, out_dim, fc=None, vbs=128):
        super().__init__()
        if fc:
            self.fc = fc
        else:
            self.fc = nn.Linear(inp_dim, out_dim * 2)
        self.bn = GBN(out_dim * 2, vbs=vbs)
        self.od = out_dim

    def forward(self, x):
        x = self.bn(self.fc(x))
        return x[:, : self.od] * torch.sigmoid(x[:, self.od :])


class FeatureTransformer(nn.Module):
    def __init__(self, inp_dim, out_dim, shared, n_ind, vbs=128):
        super().__init__()
        first = True
        self.shared = nn.ModuleList()
        if shared:
            self.shared.append(GLU(inp_dim, out_dim, shared[0], vbs=vbs))
            first = False
            for fc in shared[1:]:
                self.shared.append(GLU(out_dim, out_dim, fc, vbs=vbs))
        else:
            self.shared = None

        self.independ = nn.ModuleList()
        if first:
            self.independ.append(GLU(inp_dim, out_dim, vbs=vbs))
        for x in range(first, n_ind):
            self.independ.append(GLU(out_dim, out_dim, vbs=vbs))

        self.register_buffer("scale", torch.sqrt(torch.tensor(0.5)))

    def forward(self, x):
        if self.shared:
            x = self.shared[0](x)
            for glu in self.shared[1:]:
                x = torch.add(x, glu(x))
                x = x * self.scale
        for glu in self.independ:
            x = torch.add(x, glu(x))
            x = x * self.scale
        return x


class DecisionStep(nn.Module):
    def __init__(self, inp_dim, n_d, n_a, shared, n_ind, relax, vbs=128):
        super().__init__()
        self.fea_tran = FeatureTransformer(inp_dim, n_d + n_a, shared, n_ind, vbs)
        self.atten_tran = AttentionTransformer(n_a, inp_dim, relax, vbs)

    def forward(self, x, a, priors):
        mask, priors = self.atten_tran(a, priors)
        sparse_loss = ((-1) * mask * torch.log(mask + 1e-10)).mean()
        x = self.fea_tran(x * mask)
        return x, sparse_loss, priors


class TabNet(nn.Module):
    def __init__(
        self,
        inp_dim,
        final_out_dim,
        n_d=64,
        n_a=64,
        n_shared=2,
        n_ind=2,
        n_steps=5,
        relax=1.2,
        vbs=128,
    ):
        super().__init__()
        if n_shared > 0:
            self.shared = nn.ModuleList()
            self.shared.append(nn.Linear(inp_dim, 2 * (n_d + n_a)))
            for x in range(n_shared - 1):
                self.shared.append(nn.Linear(n_d + n_a, 2 * (n_d + n_a)))
        else:
            self.shared = None

        self.first_step = FeatureTransformer(inp_dim, n_d + n_a, self.shared, n_ind)
        self.steps = nn.ModuleList()
        for x in range(n_steps - 1):
            self.steps.append(
                DecisionStep(inp_dim, n_d, n_a, self.shared, n_ind, relax, vbs)
            )

        self.fc = nn.Linear(n_d, final_out_dim)
        self.bn = nn.BatchNorm1d(inp_dim)
        self.n_d = n_d

    def forward(self, x):
        x = self.bn(x)
        x_a = self.first_step(x)[:, self.n_d :]
        sparse_loss = torch.zeros(1, device=x.device, dtype=x.dtype)
        out = torch.zeros(x.size(0), self.n_d, device=x.device, dtype=x.dtype)
        priors = torch.ones(x.shape, device=x.device, dtype=x.dtype)

        for step in self.steps:
            x_te, l, priors = step(x, x_a, priors)
            out += F.relu(x_te[:, : self.n_d])
            x_a = x_te[:, self.n_d :]
            sparse_loss += l

        return self.fc(out), sparse_loss
