import numpy as np
import torch

def compute_var_regularization(lprobs, target):

    print(lprobs.size(), target.size())
    print(lprobs)
    neg_lprobs = lprobs * -1

    gt_mask = torch.zeros(lprobs.size()).to(device='cuda').scatter_(2, target.unsqueeze(2), 1.0)
    gt_probs = neg_lprobs * gt_mask
    gt_probs_reduced = torch.max(gt_probs, dim=-1).values
    print(gt_probs_reduced)

    regs = torch.zeros(target.size()).to(device='cuda')
    for t_i in range(1, target.size()[1]):
        regs[:, t_i] = torch.var(gt_probs_reduced[:, :t_i+1], dim=-1)
    print(regs)
    return regs

if __name__ == "__main__":

    batch_probs_1 = np.array([[0.03, 0.01, 0.04, 0.9, 0.02], [0.003, 0.002, 0.04, 0.001, 0.99], [0.6, 0.3, 0.01, 0.04, 0.05], [0.04, 0.01, 0.8, 0.06, 0.09]])
    batch_probs_2 = np.array([[0.02, 0.01, 0.95, 0.01, 0.01], [0.01, 0.9, 0.02, 0.03, 0.04], [0.05, 0.03, 0.04, 0.8, 0.08], [0.1, 0.2, 0.05, 0.04, 0.7]])
    probs = torch.tensor([batch_probs_1, batch_probs_2], device='cuda:0')
    target = torch.tensor(np.array([[3, 4, 1, 2], [2, 1, 3, 4]]), device='cuda:0')
    print(probs)
    print(target)
    lprobs = torch.log(probs)
    compute_var_regularization(lprobs, target)