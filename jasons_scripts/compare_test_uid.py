from statistics import variance, mean
import numpy as np
import scipy.stats as stats
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

def read_file(file_path):
    lines = open(file_path, 'r').readlines()
    uid_list = []
    sentence_nll_list = []
    word_nll_list = []

    for i, line in enumerate(lines):

        parts = line.replace("\n", "").split(',')
        if len(parts) != 112:
            print(i, parts)
            continue
        sentence_nll = [float(x) for x in parts]
        sentence_nll_list.append(sentence_nll)
        word_nll_list += sentence_nll
        
    sentence_nll_numpy = np.array(sentence_nll_list)
    sentence_nll_variance = np.var(sentence_nll_numpy, axis=-1)
    print(f"{file_path} - {len(sentence_nll_numpy)} var={np.mean(sentence_nll_variance):.4f}")
    return sentence_nll_variance


def compute_kendall_tau(nll_var_1, nll_var_2):
    tau, p_value = stats.kendalltau(nll_var_1, nll_var_2)
    print(f"tau={tau:4f}, p={p_value:.5f}")
    _, ax = plt.subplots()
    plt.scatter(nll_var_1, nll_var_2, linewidth=0.5)
    plt.savefig("temp.png", dpi=400)
    plt.clf()

def debug_file(f1, f2):

    lines_1 = open(f1, 'r').readlines()
    lines_1 = list(reversed(lines_1))[2:]
    lines_2 = open(f2, 'r').readlines()
    lines_2 = list(reversed(lines_2))[2:]

    for i, (line1, line2) in enumerate(zip(lines_1, lines_2)):
        if line1 != line2:
            print(i)
            print(line1, line2)
            break

if __name__ == "__main__":

    sentence_nll_variance_00 = read_file('jason-lm-test-logs-europv7-en/default_allvar00_uid_testloss.csv')
    # sentence_nll_variance_001 = read_file('jason-lm-test-logs-europv7-en/default_allvar001_uid_testloss.csv')
    # sentence_nll_variance_002 = read_file('jason-lm-test-logs-europv7-en/default_allvar002_uid_testloss.csv')
    sentence_nll_variance_003 = read_file('jason-lm-test-logs-europv7-en/default_allvar003_uid_testloss.csv')
    # compute_kendall_tau(sentence_nll_variance_00, sentence_nll_variance_00)
    # compute_kendall_tau(sentence_nll_variance_00, sentence_nll_variance_001)
    # compute_kendall_tau(sentence_nll_variance_00, sentence_nll_variance_002)
    compute_kendall_tau(sentence_nll_variance_00, sentence_nll_variance_003)