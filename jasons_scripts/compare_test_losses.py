from statistics import mean
import numpy as np
from tqdm import tqdm

#credits to http://www2.stat.duke.edu/~ar182/rr/examples-gallery/PermutationTest.html
def compute_permutation_stat(z, y):

    def run_permutation_test(pooled,sizeZ,sizeY,delta):
        np.random.shuffle(pooled)
        starZ = pooled[:sizeZ]
        starY = pooled[-sizeY:]
        return starZ.mean() - starY.mean()

    pooled = np.hstack([z, y])
    delta = z.mean() - y.mean()
    numSamples = 5000
    estimates = []
    for i in tqdm(range(numSamples)):
        estimates.append(run_permutation_test(pooled,z.size,y.size,delta))
    estimates = np.array(estimates)
    diffCount = len(np.where(estimates <= delta)[0])
    hat_asl_perm = 1.0 - (float(diffCount)/float(numSamples))
    return hat_asl_perm

def read_file(file_path):
    lines = open(file_path, 'r').readlines()
    loss_list = [float(x[:-1]) for x in lines]
    return np.array(loss_list)

if __name__ == "__main__":
    loss_list_1 = read_file("jason-lm-test-logs-wt2/dropout02_allvar003_testloss.csv") #this one should be lower (better)
    loss_list_2 = read_file("jason-lm-test-logs-wt2/dropout02_allvar00_testloss.csv") 

    hat_asl_perm = compute_permutation_stat(loss_list_2, loss_list_1)
    print(f"list_1_mean:{mean(loss_list_1):.5f}, \t list_2_mean:{mean(loss_list_2):.5f}, \t p value of (l1 < l2)={hat_asl_perm:.4f}")