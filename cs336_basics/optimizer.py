import torch
import math
from typing import Optional, Iterable, Callable
# from torch.utils.tensorboard import SummaryWriter
import os
import matplotlib.pyplot as plt

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        '''
        AdamW optimizer.
        '''
        # validate hyperparameters
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight decay value: {}".format(weight_decay))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta1 parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta2 parameter at index 1: {}".format(betas[1]))
        
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
        
    
    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                state = self.state[p]

                t = state.get('t', 1) # initialize iteration number to 0 if not present
                m = state.get('m', torch.zeros_like(p.data)) # initialize first moment of grad m to 0 if not present
                v = state.get('v', torch.zeros_like(p.data)) # initialize second moment of grad v to 0 if not present

                # p.grad is a tensor with requires_grad=True, so we need to detach it to avoid computing gradients
                # p.grad.data is a tensor with requires_grad=False
                # p.grad.data is the detached version of p.grad sharing the same data but not requiring gradients
                grad = p.grad.data # get the detached gradient of loss w.r.t. the parameter p

                m = group['betas'][0] * m + (1 - group['betas'][0]) * grad
                v = group['betas'][1] * v + (1 - group['betas'][1]) * (grad ** 2)


                # bias correction note that orignal paper use 1-indexed iteration number
                lr_t = group['lr'] * math.sqrt(1 - group['betas'][1] ** (t)) / (1 - group['betas'][0] ** (t))

                # update the parameter
                p.data -= lr_t * m / (torch.sqrt(v) + group['eps'])

                # Apply weight decay
                p.data -= group['lr'] * group['weight_decay'] * p.data

                # update iteration number
                state['t'] = t + 1

                # update m and v
                state['m'] = m
                state['v'] = v
    
        return loss
                
class CosineLR():
    def __init__(self, lr_min, lr_max, T_warmup, T_cosine):
        ''' 
        Cosine learning rate schedule.
        t: current iteration number zero-based index
        lr_min: minimum learning rate
        lr_max: maximum learning rate
        T_warmup: number of warmup iterations
        T_cosine: number of cosine iterations
        0 -> T_warmup: linear warmup -> T_cosine: cosine decay -> t > T_cosine: constant lr
        '''
        self.lr_min = lr_min
        self.lr_max = lr_max
        self.T_warmup = T_warmup
        self.T_cosine = T_cosine

    def __call__(self, t):
        if t < self.T_warmup:
            return t * self.lr_max / self.T_warmup
        elif t >= self.T_warmup and t <= self.T_cosine:
            return self.lr_min + 0.5 * (self.lr_max - self.lr_min) * (1 + math.cos(math.pi * (t - self.T_warmup) / (self.T_cosine - self.T_warmup)))
        else:
            # t > T_cosine: constant lr = lr_min
            return self.lr_min

# Note that we should make optimzer a parameter lr_schedule for safe checkpointing and loading
'''

import torch
from torch import nn
import math
 
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=1,out_channels=1,kernel_size=2,stride=1,padding=0)   
    def forward(self,x):
        out = self.conv(x)
        return out
 
net = Net()

def rule(epoch):
    lamda = math.pow(0.5, int(epoch / 3))
    return lamda
optimizer = torch.optim.SGD([{'params': net.parameters(), 'initial_lr': 0.1}], lr = 0.1)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda = rule)

for i in range(9):
    print("lr of epoch", i, "=>", scheduler.get_lr())
    optimizer.step()
    scheduler.step()
'''
class AdamWLR(torch.optim.Optimizer):
    def __init__(
            self,
            params,
            lr_schedule,
            lr=1e-3,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.01,
            last_step=-1):
        '''
        AdamW optimizer.
        '''
        # validate hyperparameters
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight decay value: {}".format(weight_decay))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta1 parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta2 parameter at index 1: {}".format(betas[1]))
        
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, last_step=last_step, lr_schedule=lr_schedule)
        super().__init__(params, defaults)
        
    
    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                state = self.state[p]

                t = state.get('t', group['last_step'] + 1) # initialize iteration number to last_step + 1 if not present
 

                m = state.get('m', torch.zeros_like(p.data)) # initialize first moment of grad m to 0 if not present
                v = state.get('v', torch.zeros_like(p.data)) # initialize second moment of grad v to 0 if not present

                # p.grad is a tensor with requires_grad=True, so we need to detach it to avoid computing gradients
                # p.grad.data is a tensor with requires_grad=False
                # p.grad.data is the detached version of p.grad sharing the same data but not requiring gradients
                grad = p.grad.data # get the detached gradient of loss w.r.t. the parameter p

                m = group['betas'][0] * m + (1 - group['betas'][0]) * grad
                v = group['betas'][1] * v + (1 - group['betas'][1]) * (grad ** 2)

                # update learning rate by lr_schedule
                group['lr'] = group['lr_schedule'](t)

                # bias correction note that orignal paper use 1-indexed iteration number in original AdamW
                # here we use 0-indexed iteration number hence t + 1
                lr_t = group['lr'] * math.sqrt(1 - group['betas'][1] ** (t + 1)) / (1 - group['betas'][0] ** (t + 1))

                # update the parameter
                p.data -= lr_t * m / (torch.sqrt(v) + group['eps'])

                # Apply weight decay
                p.data -= group['lr'] * group['weight_decay'] * p.data

                # update iteration number
                state['t'] = t + 1

                # update m and v
                state['m'] = m
                state['v'] = v
    
        return loss
                
class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=0.01):
        '''
        Stochastic Gradient Descent (SGD) optimizer.

        Args:
            params (iterable): iterable of parameters to optimize or dicts defining
                parameter groups, typically the model parameters
            lr (float, optional): learning rate (default: 0.01)
            In the case that parameters are just a single collection of torch.nn.Parameter objects, base constructor will create a single parameter group
            and assign the default hyperparameters like learning rate to it.
            We can use different hyperparameters like learning rates for different parameter groups.
        '''
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        # defaults = dict(lr=lr) # use dict() constructor to create a dictionary with the learning rate.
        defaults = {'lr': lr} # use dict literal syntax to create a dictionary with the learning rate.
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        # closure might be used to compute the loss of the model before calling optimizer.step()
        # Here we add it to comply with the base class interface. torch.optim.Optimizer
        loss = None if closure is None else closure()

        for group in self.param_groups:
            lr = group['lr'] # Get the learning rate for this group of parameters

            for p in group['params']:
                if p.grad is None:
                    continue

                # base optimizer gives us a dict self.state mapping each nn.Parameter object to a dictionary of its state.
                state = self.state[p] # Get the state of the parameter. This is a dictionary that stores the state of the parameter.
                t = state.get('t', 0) # Get iteration number for the parameter or default to 0. Safe access with get() instead of []
                grad = p.grad.data # get the gradient of loss w.r.t. the parameter p
                p.data -= lr / math.sqrt(t + 1) * grad # update the weight tensor in place
                state['t'] = t + 1 # update the iteration number for the parameter
        
        return loss

def lr_cosine_schedule(t, lr_min, lr_max, T_warmup, T_cosine):
    ''' 
    Cosine learning rate schedule.
    t: current iteration number zero-based index
    lr_min: minimum learning rate
    lr_max: maximum learning rate
    T_warmup: number of warmup iterations
    T_cosine: number of cosine iterations
    0 -> T_warmup: linear warmup -> T_cosine: cosine decay -> t > T_cosine: constant lr
    '''
    if t < T_warmup:
        return t * lr_max / T_warmup
    elif t >= T_warmup and t <= T_cosine:
        return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * (t - T_warmup) / (T_cosine - T_warmup)))
    else:
        # t > T_cosine: constant lr = lr_min
        return lr_min

def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float):
    '''
    Gradient clipping in place based on l2 norm of combined gradients.
    https://docs.pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html
    '''
    # get the total l2 norm of all parameters
    grads = [p.grad.data  for p in parameters if p.grad is not None]

    # Stack list of grad norm tensors into a single tensor and compute the l2 norm of the stacked tensor
    grad_l2_norm = torch.norm(torch.stack([g.norm(2) for g in grads]), p=2)

    if grad_l2_norm > max_l2_norm:
        for p in parameters:
            if p.grad is not None:
                p.grad.data = p.grad.data * max_l2_norm / (grad_l2_norm + 1e-6)

def test_AdamWLR():
    model = torch.nn.Linear(10, 10)
    cos_lr_min = 5e-6
    cos_lr_max = 5e-4
    cos_T_warmup = 10
    cos_T_cosine = 100

    # lr_schedule = CosineLR(cos_lr_min, cos_lr_max, cos_T_warmup, cos_T_cosine)
    # lrs = []
    # for t in range(100):
    #     print(f'lr_schedule({t}): {lr_schedule(t)}')
    #     lrs.append(lr_schedule(t))
    
    # plt.plot(lrs)
    # plt.savefig('lr_cosine_schedule.png')
    # print('Saved figure to lr_cosine_schedule.png')
    
    # # check if lr_schedule is callable
    # print(f'lr_schedule is callable: {callable(lr_schedule)}')

    optimizer = AdamWLR(model.parameters(),
                        lr_schedule=CosineLR(cos_lr_min, cos_lr_max, cos_T_warmup, cos_T_cosine),
                        lr=cos_lr_max)
    lrs = []
    for t in range(100):
        # generate a random tensor of size 10
        x = torch.randn(10)
        y = model(x)
        loss = (y - x).mean()
        loss.backward()
        print(f'loss: {loss.item()}')

        optimizer.step()
        print(optimizer.param_groups[0]['lr'])
        lrs.append(optimizer.param_groups[0]['lr'])
    
    plt.plot(lrs)
    plt.savefig('lr_cosine_schedule.png')
    print('Saved figure to lr_cosine_schedule.png')


if __name__ == "__main__":
    test_AdamWLR()
    # lr_min = 0.01
    # lr_max = 0.1
    # T_warmup = 10
    # T_cosine = 100
    # lrs = []
    # for t in range(T_warmup + T_cosine + 1):
    #     lrs.append(lr_cosine_schedule(t, lr_min, lr_max, T_warmup, T_cosine))
    

    # # save lrs per iteration to a plt figure
    # import matplotlib.pyplot as plt
    # plt.plot(lrs)
    # plt.savefig('lr_cosine_schedule.png')

# if __name__ == "__main__":
#     weights = torch.nn.Parameter(5 * torch.randn(10, 10))
#     lrs = [0.1, 1.0]
#     max_iter = 100

#     # Create tensorboard writer
#     log_dir = 'runs/sgd'
#     if os.path.exists(log_dir):
#         import shutil
#         shutil.rmtree(log_dir)  # Clear previous runs
#     writers = {lr: SummaryWriter(log_dir=f"runs/sgd/lr_{lr}") for lr in lrs}
#     writer = SummaryWriter(log_dir)

#     for lr in lrs:
#         # Reset weights for each learning rate
#         weights.data = 5 * torch.randn(10, 10)
#         opt = SGD([weights], lr=lr)
        
#         # This is the standard training loop.
#         for t in range(max_iter):
#             opt.zero_grad() # zero the gradients for all learnable parameters in optimizer
#             loss = (weights ** 2).mean() # loss is a scalar. No dim means reduction over all dimensions.
#             loss.backward() # compute the gradient of the loss w.r.t. the weights
#             opt.step() # update the weights
            
#             # Log to tensorboard
#             writer.add_scalar(f'Loss/lr_{lr}', loss.item(), t)
#             writer.add_scalar(f'Learning_Rate/lr_{lr}', opt.param_groups[0]['lr'] / math.sqrt(t + 1), t)

#             # Log to each writer
#             writers[lr].add_scalar(f'Loss/train', loss.item(), t)
        
#         print(f"Completed training with lr={lr}")

#     writer.close()
#     print(f"TensorBoard logs saved to '{log_dir}'")
#     print("To view the logs, run: tensorboard --logdir=runs")


