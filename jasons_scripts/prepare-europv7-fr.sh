#!/bin/bash
TEXT=examples/language_model/europv7-fr
fairseq-preprocess \
    --only-source \
    --nwordssrc 64000 \
    --trainpref $TEXT/europv7_fr_trainfull.txt \
    --validpref $TEXT/europv7_fr_dev2k.txt \
    --testpref $TEXT/europv7_fr_test.txt \
    --destdir data-bin/europv7-fr \
    --workers 20