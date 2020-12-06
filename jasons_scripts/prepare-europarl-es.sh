#!/bin/bash
TEXT=examples/language_model/europarl-es
fairseq-preprocess \
    --only-source \
    --trainpref $TEXT/moseseuroparlutf.es-en.es \
    --validpref $TEXT/mosesdev2006utf.es \
    --testpref $TEXT/mosesdevtest2006utf.es \
    --destdir data-bin/europarl-es-moses \
    --workers 20