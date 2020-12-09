#!/bin/bash
TEXT=examples/language_model/europarl-de
fairseq-preprocess \
    --only-source \
    --trainpref $TEXT/moseseuroparlutf.de-en.de \
    --validpref $TEXT/mosesdev2006utf.de \
    --testpref $TEXT/mosestest2006utf.de \
    --destdir data-bin/europarl-de-moses \
    --workers 20