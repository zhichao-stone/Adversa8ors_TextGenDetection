import torch
import torch.nn.functional as F



class MPULoss:
    def __init__(self, 
        pi: float = 0.2, 
        max_length: int = 512, 
        lamb: float = 0.4, 
        device: torch.device = torch.device("cuda")
    ) -> None:
        self.pi = pi
        self.max_length = max_length
        self.lamb = lamb
        self.device = device
        self.loss_fn = lambda x: F.sigmoid(-x)

        # compute expectation pi for each length
        expectations = [self.get_expectation(i, pi) for i in range(0, max_length+1)]
        self.prior = torch.stack(expectations)
        print("All dynamic priors calculated.")

    def get_expectation(self, length: int, pi: float) -> torch.Tensor:
        device = self.device

        if length < 3:
            return torch.tensor(pi).float().to(device)
        
        dim = length+1
        state = torch.zeros((1, dim)).to(device)
        state[0, 0] = 1.

        P = torch.zeros((dim, dim)).to(device)
        P[1:, :-1] += pi * torch.eye(length).to(device)
        P[:-1, 1:] += (1-pi) * torch.eye(length).to(device)
        P[0, 0] += pi
        P[length, length] += (1 - pi)

        trans = torch.eye(dim).to(device)
        for _ in range(length):
            trans = torch.matmul(trans, P)
        distribution = torch.matmul(state, trans).squeeze(0)
        expectation = 1 - ((distribution * torch.arange(0, dim).to(device)).sum() / length)
        return expectation.to(device)

    def __call__(self, input: torch.Tensor, label: torch.Tensor, length: int) -> torch.Tensor:
        prior = self.prior[length]
        
        positive_x = (label == 1).float()
        unlabel_x = (label == -1).float()
        n_positive = torch.max(torch.sum(positive_x), torch.tensor(1.0).to(self.device))
        n_unlabel = torch.max(torch.sum(unlabel_x), torch.tensor(1.0).to(self.device))
        positive_y = self.loss_fn(input)
        unlabel_y = self.loss_fn(- input)

        ploss = torch.sum(prior * positive_x / n_positive * positive_y.squeeze())
        nloss = torch.sum((unlabel_x / n_unlabel - prior * positive_x / n_positive) * unlabel_y.squeeze())
        
        nloss = torch.max(torch.tensor(0.0).to(self.device), nloss)
        loss = ploss + nloss

        return self.lamb * loss