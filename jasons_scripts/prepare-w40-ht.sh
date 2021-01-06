#!/bin/bash
TEXT=examples/language_model/mwiki-ht
fairseq-preprocess \
    --only-source \
    --nwordssrc 128000 \
    --trainpref $TEXT/htmoses_train.txt \
    --validpref $TEXT/htmoses_dev.txt \
    --testpref $TEXT/htmoses_testfull.txt \
    --destdir data-bin/mwiki_ht \
    --workers 20