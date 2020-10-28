import math

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


class Linear(nn.Module):
    def __init__(self, in_dims: int, out_dims: int, activation: str = None, batchnorm: bool = False):
        super(Linear, self).__init__()
        self.net = nn.Linear(in_dims, out_dims, bias=True)
        if batchnorm:
            self.bn = nn.BatchNorm1d(out_dims)
        else:
            self.bn = nn.Identity()
        if activation is None:
            self.act = nn.Identity()
        elif activation == 'relu':
            self.act = nn.ReLU(inplace=True)
        else:
            raise NotImplementedError

    def forward(self, x):
        return self.act(self.bn(self.net(x)))


class OddOneOutHead(nn.Module):
    def __init__(self, embedding_size: int, num_embeddings: int, method: str, base_dims: int = 128,
                 batchnorm: bool = True):
        super(OddOneOutHead, self).__init__()
        self.embedding_size = embedding_size
        self.num_embedding = num_embeddings

        self.method = method
        if 'sod' in self.method:
            in_dims = embedding_size
        elif self.method == 'concat':
            in_dims = embedding_size * num_embeddings
        else:
            raise NotImplementedError

        net = [Linear(in_dims=in_dims, out_dims=base_dims, activation='relu', batchnorm=batchnorm),
               Linear(in_dims=base_dims, out_dims=base_dims, activation='relu', batchnorm=batchnorm),
               Linear(in_dims=base_dims, out_dims=num_embeddings, activation=None, batchnorm=False)]
        self.net = nn.Sequential(*net)

        idxs1 = np.concatenate([np.ones(num_embeddings - i - 1) * (num_embeddings - i - 1)
                                for i in range(num_embeddings-1)], axis=0).reshape(-1)
        idxs2 = np.concatenate([np.arange(num_embeddings - i - 1) for i in range(num_embeddings-1)])

        self.idxs1 = nn.Parameter(torch.from_numpy(idxs1).long(), requires_grad=False)
        self.idxs2 = nn.Parameter(torch.from_numpy(idxs2).long(), requires_grad=False)

    def forward(self, x):
        batchsize = x.shape[0]
        assert x.shape[1] == self.num_embedding and x.shape[2] == self.embedding_size
        if self.method == 'cosine-sod':
            x = F.normalize(x, dim=2)  # normalize to get cosine embeddings

        if 'sod' in self.method:
            embeddings1 = x[:, self.idxs1, :]
            print(self.idxs1)
            print(self.idxs2)
            embeddings2 = x[:, self.idxs2, :]
            differences = embeddings1 - embeddings2  # B x N x D
            out = torch.sum(differences, dim=1)  # B x D
        elif self.method == 'concat':
            out = x.view(batchsize, -1)
        else:
            raise NotImplementedError
        out = self.net(out)
        return out


def main():
    model = OddOneOutHead(embedding_size=32, num_embeddings=6, method='cosine-sod')
    data = torch.randn(size=[10, 6, 32])
    out = model(data)
    print(out.shape)


if __name__ == '__main__':
    main()
