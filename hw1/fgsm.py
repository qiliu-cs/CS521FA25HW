import torch
import torch.nn as nn


# fix seed so that random initialization always performs the same 
torch.manual_seed(13)


# create the model N as described in the question
N = nn.Sequential(nn.Linear(10, 10, bias=False),
                  nn.ReLU(),
                  nn.Linear(10, 10, bias=False),
                  nn.ReLU(),
                  nn.Linear(10, 3, bias=False))

# random input
x = torch.rand((1,10)) # the first dimension is the batch size; the following dimensions the actual dimension of the data
x.requires_grad_() # this is required so we can compute the gradient w.r.t x

# problem 1
print("Problem 1")

t = 0 # target class

epsReal = 0.5  #depending on your data this might be large or small
eps = epsReal - 1e-7 # small constant to offset floating-point erros

# The network N classfies x as belonging to class 2
original_class = N(x).argmax(dim=1).item()  # TO LEARN: make sure you understand this expression
print("Original Class: ", original_class)
assert(original_class == 2)

# compute gradient
# note that CrossEntropyLoss() combines the cross-entropy loss and an implicit softmax function
L = nn.CrossEntropyLoss()
loss = L(N(x), torch.tensor([t], dtype=torch.long)) # TO LEARN: make sure you understand this line
loss.backward()

# your code here
# adv_x should be computed from x according to the fgsm-style perturbation such that the new class of xBar is the target class t above
# hint: you can compute the gradient of the loss w.r.t to x as x.grad
adv_x = x - eps * torch.sign(x.grad)

new_class = N(adv_x).argmax(dim=1).item()
print("New Class: ", new_class)
assert(new_class == t)
# it is not enough that adv_x is classified as t. We also need to make sure it is 'close' to the original x. 
print(torch.norm((x-adv_x),  p=float('inf')).data)
assert( torch.norm((x-adv_x), p=float('inf')) <= epsReal)






# problem 2
print("\n\nProblem 2")

# Reset and use same seed to get same initial input
torch.manual_seed(13)
x2 = torch.rand((1,10))
x2.requires_grad_()

t2 = 1  # target class for problem 2
original_class2 = N(x2).argmax(dim=1).item()
print("Original Class: ", original_class2)

# Iterative FGSM attack
print("Using Iterative FGSM:")
x2_iter = x2.clone().detach()
alpha = 0.05  # step size
iterations = 50
eps_total = 2.0

for i in range(iterations):
    x2_iter_temp = x2_iter.clone().detach().requires_grad_(True)
    
    L2 = nn.CrossEntropyLoss()
    loss2 = L2(N(x2_iter_temp), torch.tensor([t2], dtype=torch.long))
    loss2.backward()
    
    # Take a small step
    with torch.no_grad():
        perturbation = -alpha * torch.sign(x2_iter_temp.grad)
        x2_iter = x2_iter + perturbation
        
        # Project back to constraint set
        total_perturbation = x2_iter - x2
        total_perturbation = torch.clamp(total_perturbation, -eps_total, eps_total)
        x2_iter = x2 + total_perturbation
    
    current_class = N(x2_iter).argmax(dim=1).item()
    if current_class == t2:
        perturbation_norm_iter = torch.norm(total_perturbation, p=float('inf')).item()
        print(f"I-FGSM successful after {i+1} iterations!")
        print(f"New Class: {current_class}")
        print(f"L-inf norm: {perturbation_norm_iter:.6f}")
        
        # Final verification
        with torch.no_grad():
            final_logits = N(x2_iter)
            print(f"Final logits: {final_logits.squeeze()}")
        
        adv_x2 = x2_iter
        new_class2 = current_class
        break