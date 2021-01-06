#!/bin/bash
TEXT=examples/language_model/mwiki-cv
fairseq-preprocess \
    --only-source \
    --nwordssrc 128000 \
    --trainpref $TEXT/cvmoses_train.txt \
    --validpref $TEXT/cvmoses_dev.txt \
    --testpref $TEXT/cvmoses_testfull.txt \
    --destdir data-bin/mwiki_cv \
    --workers 20