#!/bin/bash
TEXT=examples/language_model/m_wiki_sw
fairseq-preprocess \
    --only-source \
    --nwordssrc 128000 \
    --trainpref $TEXT/swmoses_train.txt \
    --validpref $TEXT/swmoses_dev.txt \
    --testpref $TEXT/swmoses_test.txt \
    --destdir data-bin/mwiki_sw \
    --workers 20