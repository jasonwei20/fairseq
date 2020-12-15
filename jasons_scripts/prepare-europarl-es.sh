#!/bin/bash
TEXT=examples/language_model/europarl-es
fairseq-preprocess \
    --only-source \
    --nwordssrc 64000 \
    --trainpref $TEXT/moseseuroparlutf.es-en.es \
    --validpref $TEXT/mosesdev2006utf.es \
    --testpref $TEXT/mosestest2006utf.es \
    --destdir data-bin/europarl-es-moses \
    --workers 20