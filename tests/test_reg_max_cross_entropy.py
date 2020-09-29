import numpy as np
import torch

def compute_max_regularization_slow(lprobs, target):

    print(lprobs.size(), target.size())
    print(lprobs)
    neg_lprobs = lprobs * -1
    regs = torch.empty(target.size()) #regularizer is the same shape as the target (B x T)

    for b in range(regs.size()[0]):
        for t in range(regs.size()[1]):
            max_candidates = []
            for t_i in range(0, t+1):
                gt_index = target[b, t_i]
                max_candidate = lprobs[b, t_i, gt_index] * -1
                max_candidates.append(max_candidate)
            reg_value = max(max_candidates)
            regs[b, t] = reg_value

    print(regs)
    return regs

def compute_max_regularization_faster(lprobs, target):

    print(lprobs.size(), target.size())
    print(lprobs)
    neg_lprobs = lprobs * -1

    gt_mask = torch.zeros(lprobs.size()).to(device='cuda').scatter_(2, target.unsqueeze(2), 1.0)
    gt_probs = neg_lprobs * gt_mask
    max_probs = torch.max(gt_probs, dim=-1).values

    regs = torch.empty(target.size()).to(device='cuda')
    for t_i in range(0, target.size()[1]):
        regs[:, t_i] = torch.max(max_probs[:, :t_i+1], -1).values
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
    compute_max_regularization_slow(lprobs, target)
    compute_max_regularization_faster(lprobs, target)