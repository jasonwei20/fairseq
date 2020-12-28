#!/bin/bash
TEXT=examples/language_model/w40-tl
fairseq-preprocess \
    --only-source \
    --nwordssrc 128000 \
    --trainpref $TEXT/tlmoses_wiki_train.txt \
    --validpref $TEXT/tlmoses_wiki_dev.txt \
    --testpref $TEXT/tlmoses_wiki_test.txt \
    --destdir data-bin/w40-tl \
    --workers 20