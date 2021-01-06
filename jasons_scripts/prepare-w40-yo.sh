#!/bin/bash
TEXT=examples/language_model/mwiki-yo
fairseq-preprocess \
    --only-source \
    --nwordssrc 128000 \
    --trainpref $TEXT/yomoses_train.txt \
    --validpref $TEXT/yomoses_dev.txt \
    --testpref $TEXT/yomoses_test.txt \
    --destdir data-bin/mwiki_yo \
    --workers 20