# Running Wikitext-103 Experiments with UID Regularizers

Clone my repository:
```
git clone https://github.com/jasonwei20/fairseq.git
cd fairseq
```

Install the environment (this is what works for me on leonhard):
```
module load gcc/6.3.0 python_gpu/3.8.5 hdf5 eth_proxy
python3 -m venv envapex
source envapex/bin/activate
PYTHONPATH=$(which python)
$PYTHONPATH -m pip install torch==1.4.0 torchvision==0.5.0
$PYTHONPATH -m pip install --editable ./
$PYTHONPATH -m pip install matplotlib
```

Optionally, install apex (takes a long time for me)
```
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" \
  --global-option="--deprecated_fused_adam" --global-option="--xentropy" \
  --global-option="--fast_multihead_attn" ./
```

Download and prepare the WikiText-103 dataset:
```
cd examples/language_model/
bash prepare-wikitext-103.sh
cd ../..
TEXT=examples/language_model/wikitext-103
fairseq-preprocess \
    --only-source \
    --trainpref $TEXT/wiki.train.tokens \
    --validpref $TEXT/wiki.valid.tokens \
    --testpref $TEXT/wiki.test.tokens \
    --destdir data-bin/wikitext-103 \
    --workers 20
```

Train the model with different beta coefficients. Gotta change a couple of things:
- `--save-dir`: where to save the checkpoints (only keeping the best)
- `--jason-log-dir`: where to save the logging (i.e., the results)
- Based on the memory of your GPU, change `--max-tokens` and `--update-freq` accordingly. 
- Make sure `--beta-coefficient` is what you want. 

```
bsub -I -R "rusage[mem=32000, ngpus_excl_p=1]" \
fairseq-train --task language_modeling \
data-bin/wikitext-103 \
--save-dir /cluster/scratch//jaswei/checkpoints-lm-wt103/asdf_allvar003 \
--jason-log-dir jason-lm-logs-wt103/asdf_allvar003 \
--arch transformer_lm --share-decoder-input-output-embed \
--dropout 0.1 \
--optimizer adam --adam-betas '(0.9, 0.98)' --weight-decay 0.01 --clip-norm 0.0 \
--lr 0.0005 --lr-scheduler inverse_sqrt --warmup-updates 4000 --warmup-init-lr 1e-07 \
--tokens-per-sample 512 --sample-break-mode none \
--max-tokens 2048 --update-freq 16 \
--max-update 300000 \
--fp16 \
--no-epoch-checkpoints --no-last-checkpoints \
--criterion reg_allvar_cross_entropy --beta-coefficient 0.03 \
```

Outputs:
In the logging directory (`--jason-log-dir`), you'll find an image (`loss.png`) which has the training plots and, in the title, the lowest dev loss.