#!/bin/bash
TEXT=examples/language_model/w40-et
fairseq-preprocess \
    --only-source \
    --nwordssrc 256000 \
    --trainpref $TEXT/etmoses_wiki_train.txt \
    --validpref $TEXT/etmoses_wiki_dev.txt \
    --testpref $TEXT/etmoses_wiki_test.txt \
    --destdir data-bin/w40-et \
    --workers 20