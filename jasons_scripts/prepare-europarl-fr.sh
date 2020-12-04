#!/bin/bash
TEXT=examples/language_model/europarl-fr
fairseq-preprocess \
    --only-source \
    --trainpref $TEXT/moseseuroparlutf.fr.srilm \
    --validpref $TEXT/mosesdev2006utf.fr \
    --testpref $TEXT/mosesdevtest2006utf.fr \
    --destdir data-bin/europarl-fr-moses \
    --workers 20