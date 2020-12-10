#!/bin/bash
TEXT=examples/language_model/w40-tr
fairseq-preprocess \
    --only-source \
    --nwordssrc 64000 \
    --trainpref $TEXT/trmoses_wiki_train.txt \
    --validpref $TEXT/trmoses_wiki_dev.txt \
    --testpref $TEXT/trmoses_wiki_test.txt \
    --destdir data-bin/w40-tr \
    --workers 20