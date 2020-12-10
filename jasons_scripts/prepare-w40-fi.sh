#!/bin/bash
TEXT=examples/language_model/w40-fi
fairseq-preprocess \
    --only-source \
    --nwordssrc 64000 \
    --trainpref $TEXT/fimoses_wiki_train.txt \
    --validpref $TEXT/fimoses_wiki_dev.txt \
    --testpref $TEXT/fimoses_wiki_test.txt \
    --destdir data-bin/w40-fi \
    --workers 20