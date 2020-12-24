#!/bin/bash
TEXT=examples/language_model/europv7-de
fairseq-preprocess \
    --only-source \
    --nwordssrc 64000 \
    --trainpref $TEXT/europv7_de_trainfull.txt \
    --validpref $TEXT/europv7_de_dev2k.txt \
    --testpref $TEXT/europv7_de_test.txt \
    --destdir data-bin/europv7-de \
    --workers 20