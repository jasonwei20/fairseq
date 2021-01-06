#!/bin/bash
TEXT=examples/language_model/europv7-cs
fairseq-preprocess \
    --only-source \
    --nwordssrc 64000 \
    --trainpref $TEXT/europv7_cs_trainfull.txt \
    --validpref $TEXT/europv7_cs_dev2k.txt \
    --testpref $TEXT/europv7_cs_test.txt \
    --destdir data-bin/europv7-cs \
    --workers 20