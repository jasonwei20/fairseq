#!/bin/bash
TEXT=examples/language_model/m_wiki_uz
fairseq-preprocess \
    --only-source \
    --nwordssrc 128000 \
    --trainpref $TEXT/uzmoses_train.txt \
    --validpref $TEXT/uzmoses_dev.txt \
    --testpref $TEXT/uzmoses_test.txt \
    --destdir data-bin/mwiki_uz \
    --workers 20