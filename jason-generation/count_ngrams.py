from statistics import mean

def get_ngrams(line, n):
    line = line[:-1].lower()
    line = line.replace("-", " ")
    line = ''.join([c for c in line if c in 'qwertyuiopasdfghjklzxcvbnm '])
    words = line.split(' ')
    if len(words) < n:
        return []
    elif len(words) == n:
        return [' '.join(words)]
    else:
        ngrams = []
        for start_idx in range(len(words) - n):
            end_idx = start_idx + n
            ngram_words = words[start_idx:end_idx]
            ngrams.append(' '.join(ngram_words))
        return ngrams

def count_ngrams_sent_avg(file_path):
    print(f"\n{file_path}")
    lines = open(file_path, 'r').readlines()
    
    for n in range(1, 5):
        
        sent_level_percent_unique_list = []

        for line in lines:
            ngrams = get_ngrams(line, n)
            if len(ngrams) > 5:
                sent_level_percent_unique = len(set(ngrams)) / len(ngrams)
                sent_level_percent_unique_list.append(sent_level_percent_unique)
                if n == 4 and sent_level_percent_unique < 1:
                    print(sent_level_percent_unique, line)

        print(f"% unique {n}-grams (averaged over generated sample) = {mean(sent_level_percent_unique_list):.4f}")


def count_ngrams(file_path):
    print(f"\n{file_path}")
    lines = open(file_path, 'r').readlines()

    for n in range(2, 4):
        
        all_ngrams = []
        for line in lines:
            ngrams = get_ngrams(line, n)
            all_ngrams += ngrams
        
        num_ngrams = len(all_ngrams)
        num_unique_ngrams = len(set(all_ngrams))

        print(f"{num_unique_ngrams} unique of {num_ngrams} {n}-grams; % unique {n}-grams = {num_unique_ngrams/num_ngrams:.4f}")

if __name__ == "__main__":

    # count_ngrams('generated-outputs/allvar00-checkpoint_words.txt')
    # count_ngrams('generated-outputs/europarl-en/allvar0006-checkpoint_words.txt')
    count_ngrams('generated-outputs/europarl-en/allvar0008-checkpoint_words.txt')
    # count_ngrams('generated-outputs/allvar001-checkpoint_words.txt')
    # count_ngrams('generated-outputs/allvar002-checkpoint_words.txt')
    # count_ngrams('generated-outputs/allvar003-checkpoint_words.txt')
    # count_ngrams('generated-outputs/europarl-en/allvar005-checkpoint_words.txt')

    # count_ngrams('generated-outputs/europarl-en/local0006-checkpoint_words.txt')
    # count_ngrams('generated-outputs/local0008-checkpoint_words.txt')
    # count_ngrams('generated-outputs/local001-checkpoint_words.txt')
    count_ngrams('generated-outputs/europarl-en/local002-checkpoint_words.txt')
    # count_ngrams('generated-outputs/local003-checkpoint_words.txt')
    # count_ngrams('generated-outputs/europarl-en/local005-checkpoint_words.txt')

    # count_ngrams('generated-outputs/europarl-en/allvar005-checkpoint-best_words.txt')
    # count_ngrams('generated-outputs/europarl-en/allvar003-checkpoint-best_words.txt')
    # count_ngrams('generated-outputs/europarl-en/allvar001-checkpoint-best_words.txt')

    # count_ngrams('generated-outputs/europarl-en/local0003-checkpoint-best_words.txt')
    # count_ngrams('generated-outputs/europarl-en/local001-checkpoint-best_words.txt')
    # count_ngrams('generated-outputs/europarl-en/local003-checkpoint-best_words.txt')

    # count_ngrams('generated-outputs/europarl-de/allvar00-checkpoint-best_words.txt')
    # count_ngrams('generated-outputs/europarl-de/allvar003-checkpoint-best_words.txt')
    # count_ngrams('generated-outputs/europarl-de/local002-checkpoint-best_words.txt')

    # count_ngrams('generated-outputs/europarl-en/allvar00-checkpoint-best_words.txt')
    # count_ngrams('generated-outputs/europarl-en/allvar003-checkpoint-best_words.txt')
    # count_ngrams('generated-outputs/europarl-en/local002-checkpoint-best_words.txt')

    # count_ngrams_sent_avg('generated-outputs/europarl-en/allvar00-checkpoint-best_words.txt')
    # count_ngrams_sent_avg('generated-outputs/europarl-en/allvar003-checkpoint-best_words.txt')