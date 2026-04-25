import torch
import torch.nn as nn
import torch.nn.functional as F

class MemoryMLP(nn.Module):
    def __init__(self, d_model: int, d_inner: int):
        super().__init__()
        self.W1 = nn.Parameter(torch.empty(d_inner, d_model))
        self.W2 = nn.Parameter(torch.empty(d_model, d_inner))
        self.b1 = nn.Parameter(torch.zeros(d_inner))
        self.b2 = nn.Parameter(torch.zeros(d_model))
        nn.init.xavier_uniform_(self.W1)
        nn.init.xavier_uniform_(self.W2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.gelu(F.linear(x, self.W1, self.b1))
        return F.linear(h, self.W2, self.b2)

class TitansMemory(nn.Module):
    def __init__(self, d_model: int, lr=0.01, momentum=0.9, weight_decay=0.01):
        super().__init__()
        self.d_model = d_model
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        # Inner model: 4x expansion is standard for high capacity
        self.memory = MemoryMLP(d_model, d_model * 4)

    def _update_memory(self, loss, momentum_buffers):
        weights = [self.memory.W1, self.memory.W2, self.memory.b1, self.memory.b2]
        
        # Robust Gradient Computation
        grads = torch.autograd.grad(
            loss, weights, 
            create_graph=self.training, # Only keep graph if we are training the outer model
            retain_graph=True
        )

        with torch.no_grad():
            for i, (w, g) in enumerate(zip(weights, grads)):
                # Momentum update
                momentum_buffers[i] = self.momentum * momentum_buffers[i] + g
                # Update weights with weight decay (Forgetting mechanism)
                w.copy_(w - self.lr * (momentum_buffers[i] + self.weight_decay * w))

    def forward(self, x: torch.Tensor):
        # x: (batch, seq_len, d_model)
        batch_size, seq_len, _ = x.shape
        device = x.device
        
        # Ensure memory is on the same device as input
        self.memory.to(device)
        
        # Initialize momentum buffers on the correct device
        momentum_buffers = [torch.zeros_like(p).to(device) for p in self.memory.parameters()]
        
        outputs = []
        for t in range(seq_len):
            token = x[:, t, :] # (batch, d_model)
            
            # Step 1: Retrieve from existing weights
            with torch.no_grad():
                outputs.append(self.memory(token))
            
            # Step 2: Compute surprise (MSE)
            # We treat the input as the target for reconstruction
            loss = F.mse_loss(self.memory(token), token.detach())
            
            # Step 3: Test-Time Training update
            self._update_memory(loss, momentum_buffers)
            
        return torch.stack(outputs, dim=1)
