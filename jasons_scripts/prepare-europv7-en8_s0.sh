#!/bin/bash
TEXT=examples/language_model/europv7-en
fairseq-preprocess \
    --only-source \
    --nwordssrc 64000 \
    --trainpref $TEXT/europv7_en_train8_s0.txt \
    --validpref $TEXT/europv7_en_dev2k.txt \
    --testpref $TEXT/europv7_en_test.txt \
    --destdir data-bin/europv7-en8_s0 \
    --workers 20