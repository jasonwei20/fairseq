#!/bin/bash
TEXT=examples/language_model/w40-id
fairseq-preprocess \
    --only-source \
    --nwordssrc 128000 \
    --trainpref $TEXT/idmoses_wiki_train.txt \
    --validpref $TEXT/idmoses_wiki_dev.txt \
    --testpref $TEXT/idmoses_wiki_test.txt \
    --destdir data-bin/w40-id \
    --workers 20