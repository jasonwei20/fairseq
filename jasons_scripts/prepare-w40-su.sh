#!/bin/bash
TEXT=examples/language_model/mwiki-su
fairseq-preprocess \
    --only-source \
    --nwordssrc 128000 \
    --trainpref $TEXT/sumoses_train.txt \
    --validpref $TEXT/sumoses_dev.txt \
    --testpref $TEXT/sumoses_testfull.txt \
    --destdir data-bin/mwiki_su \
    --workers 20