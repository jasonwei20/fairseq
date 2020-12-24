#!/bin/bash
TEXT=examples/language_model/europv7-es
fairseq-preprocess \
    --only-source \
    --nwordssrc 64000 \
    --trainpref $TEXT/europv7_es_trainfull.txt \
    --validpref $TEXT/europv7_es_dev2k.txt \
    --testpref $TEXT/europv7_es_test.txt \
    --destdir data-bin/europv7-es \
    --workers 20