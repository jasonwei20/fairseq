from statistics import mean
import numpy as np
from tqdm import tqdm

def read_file(file_path):
    lines = open(file_path, 'r').readlines()
    loss_list = [float(x[:-1]) for x in lines]
    return np.array(loss_list)

def get_file_mean(file_path):
    loss_list = read_file(file_path)
    return mean(loss_list)

if __name__ == "__main__":

    loss_file_list = [
        # "tmp/jason-lm-test-logs-europv7-en2/default_allvar00_testloss.csv",
        # "tmp/jason-lm-test-logs-europv7-en2/default_allvar002_testloss.csv",
        # "tmp/jason-lm-test-logs-europv7-en2/default_local001_testloss.csv",
        # "tmp/jason-lm-test-logs-europv7-en4/default_allvar00_testloss.csv",
        # "tmp/jason-lm-test-logs-europv7-en4/default_allvar002_testloss.csv",
        # "tmp/jason-lm-test-logs-europv7-en4/default_local001_testloss.csv",
        # "tmp/jason-lm-test-logs-europv7-en8/default_allvar00_testloss.csv",
        # "tmp/jason-lm-test-logs-europv7-en8/default_allvar002_testloss.csv",
        # "tmp/jason-lm-test-logs-europv7-en8/default_local0007_testloss.csv",
        # "tmp/jason-lm-test-logs-europv7-en12/default_allvar00_testloss.csv",
        # "tmp/jason-lm-test-logs-europv7-en12/default_allvar002_testloss.csv",
        # "tmp/jason-lm-test-logs-europv7-en12/default_local0007_testloss.csv",
        # "tmp/jason-lm-test-logs-europv7-en16/default_allvar00_testloss.csv",
        # "tmp/jason-lm-test-logs-europv7-en16/default_allvar005_testloss.csv",
        # "tmp/jason-lm-test-logs-europv7-en16/default_local002_testloss.csv",
        # "tmp/jason-lm-test-logs-europv7-en24/default_allvar00_testloss.csv",
        # "tmp/jason-lm-test-logs-europv7-en24/default_allvar004_testloss.csv",
        # "tmp/jason-lm-test-logs-europv7-en24/default_local002_testloss.csv",
        # "tmp/jason-lm-test-logs-europv7-en32/default_allvar00_testloss.csv",
        # "tmp/jason-lm-test-logs-europv7-en32/default_allvar001_testloss.csv",
        # "tmp/jason-lm-test-logs-europv7-en32/default_local001_testloss.csv",
        # "tmp/jason-lm-test-logs-europv7-en2_s0/default_allvar00_testloss.csv",
        # "tmp/jason-lm-test-logs-europv7-en2_s0/default_allvar002_testloss.csv",
        # "tmp/jason-lm-test-logs-europv7-en2_s0/default_local001_testloss.csv",
        # "tmp/jason-lm-test-logs-europv7-en8_s0/default_allvar00_testloss.csv",
        # "tmp/jason-lm-test-logs-europv7-en8_s0/default_allvar002_testloss.csv",
        # "tmp/jason-lm-test-logs-europv7-en8_s0/default_local0008_testloss.csv",
        # "tmp/jason-lm-test-logs-europv7-en2/default_local001_testloss.csv"
        "tmp/jason-lm-test-logs-w40-fi/default_allvar00_testloss.csv",
        "tmp/jason-lm-test-logs-w40-fi/default_allvar001_testloss.csv",
        "tmp/jason-lm-test-logs-w40-fi/default_local0006_testloss.csv",
    ]
    for loss_file in loss_file_list:
        file_mean = get_file_mean(loss_file)
        print(f"{loss_file}:\t {file_mean:.5f}")
