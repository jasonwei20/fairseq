from fairseq.models.transformer_lm import TransformerLanguageModel
from tqdm import tqdm
import sys
from statistics import mean, stdev

def write_and_print(writer, line):
    print(line)
    writer.write(line)

checkpoint_folder = sys.argv[1] #'generation-checkpoints/europarl-en'
input_checkpoint = sys.argv[2] #'allvar00-checkpoint-best.pt'
out_words = 'generated-outputs/' + checkpoint_folder.split('/')[-1] + "/" + input_checkpoint.split('.')[0] + '_words.txt'
out_probs = 'generated-outputs/' + checkpoint_folder.split('/')[-1] + "/" + input_checkpoint.split('.')[0] + '_probs.txt'
out_logs = 'generated-outputs/' + checkpoint_folder.split('/')[-1] + "/" + input_checkpoint.split('.')[0] + '_logs.txt'
start = 0
num = 10000
custom_lm = TransformerLanguageModel.from_pretrained(checkpoint_folder, input_checkpoint, tokenizer='moses')
custom_lm = custom_lm.eval()
custom_lm.cuda()

num_lines = 0
try:
    with open(out_words, 'r') as f:
        num_lines = len(f.read().splitlines())
except IOError:
    pass

big_log_prob_list = []
seq_log_probs = []
seq_len_list = []

with open(out_words, 'a' if num_lines > 0 else 'w') as f:
    with open(out_probs, 'a' if num_lines > 0 else 'w') as f_probs:
        for i in tqdm(range(start+num_lines, start+num)):
            line, log_probs = custom_lm.sample('',
                seed=i, 
                unnormalized=True, 
                max_len_b=512, 
                sampling=True, 
                beam=1
            )
            log_probs = [float(x) for x in log_probs]
            log_probs_line = ','.join([f"{x:.5f}" for x in log_probs])
            big_log_prob_list += log_probs
            seq_nll = sum(log_probs)
            seq_log_probs.append(seq_nll)
            seq_len = len(log_probs)
            seq_len_list.append(seq_len)

            f.write(f"{line}\n")
            f_probs.write(f"{log_probs_line}\n")

    # print(time.time()-t)

mean_lp = mean(big_log_prob_list)
stdev_lp = stdev(big_log_prob_list)
mean_seq_len = mean(seq_len_list)
stdev_seq_len = stdev(seq_len_list)
mean_seq_nll = mean(seq_log_probs)
stdev_seq_nll = stdev(seq_log_probs)

with open(out_logs, 'w') as writer:
    write_and_print(writer, f"mean log prob:{mean_lp:.4f}\n")
    write_and_print(writer, f"stdev log prob:{stdev_lp:.4f}\n")
    write_and_print(writer, f"mean seq len:{mean_seq_len:.4f}\n")
    write_and_print(writer, f"stdev seq len:{stdev_seq_len:.4f}\n")
    write_and_print(writer, f"mean seq nll:{mean_seq_nll:.4f}\n")
    write_and_print(writer, f"stdev seq nll:{stdev_seq_nll:.4f}\n")