import torch
import torch.nn as nn


class MMSBUserCluster(nn.Module):

    def __init__(self, K, input_dim, hidden_dim) -> None:
        """
        @param K: number of clusters
        @param hidden_dim: dimension of hidden layer
        @param input_dim: dimension of input layer, which is the number of projects
        
        @return: None
        """
        super().__init__()

        self.alpha = nn.parameter.Parameter(torch.ones(K))

        self.K = K
        
        # block interaction possibility matrix
        self.B = nn.parameter.Parameter(torch.rand(K, K))
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, K)
        )

        self.decoder = nn.Sequential(
            nn.Linear(K, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

        self.recon_loss = torch.nn.MSELoss()


    def forward(self, S):
        """
        @param S: the adjacency matrix of the user-project bipartite graph, shape: (N, M), the value is success rate
        
        @return: the loss of the model, including the reconstruction loss and the MMD loss
        """
        # （N, M）-> (N, K)
        q_pi = self.encoder(S)

        # (N, K) -> (N, M)
        S_bar = self.decoder(q_pi)        

        recon_loss = self.recon_loss(S, S_bar)

        # MMD loss
        mmd_loss = self._mmd_loss(q_pi)

        return recon_loss + mmd_loss
    
    def _mmd_loss(self, q_pi):
        """
        @param q_pi: the probability of each user belonging to each cluster, shape: (N, K)
        
        @return: the MMD loss, the proir of the cluster is Dirichlet distribution with parameter self.alpha
        """
        # (N, K) -> (N, 1, K)
        q_pi = q_pi.unsqueeze(1)

        # (N, 1, K) -> (N, K, 1)
        q_pi_t = q_pi.transpose(1, 2)

        # (N, K, 1) * (1, K, K) -> (N, K, K)
        q_pi_pi_t = torch.matmul(q_pi, q_pi_t)

        # (N, K, K) * (K, K) -> (N, K, K)
        q_pi_pi_t = q_pi_pi_t * self.B

        # (N, K, K) -> (N, K)
        q_pi_pi_t = torch.sum(q_pi_pi_t, dim=2)

        # (N, K) -> (N, K)
        q_pi_pi_t = q_pi_pi_t * self.alpha

        # (N, K) -> (N, 1)
        q_pi_pi_t = torch.sum(q_pi_pi_t, dim=1).unsqueeze(1)

        # (N, K) -> (N, K)
        q_pi = q_pi * self.alpha

        # (N, K) -> (N, 1)
        q_pi = torch.sum(q_pi, dim=1).unsqueeze(1)

        # (N, 1) -> (N, 1)
        q_pi_pi_t = q_pi_pi_t / q_pi

        # (N, 1) -> (N, 1)
        q_pi_pi_t = torch.log(q_pi_pi_t)

        # (N, 1) -> (1, 1)
        q_pi_pi_t = torch.mean(q_pi_pi_t)

        mmd_loss = -q_pi_pi_t
        return mmd_loss




