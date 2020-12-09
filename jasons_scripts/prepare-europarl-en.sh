#!/bin/bash
TEXT=examples/language_model/europarl-en
fairseq-preprocess \
    --only-source \
    --trainpref $TEXT/moseseuroparlutf.de-en.en \
    --validpref $TEXT/mosesdev2006utf.en \
    --testpref $TEXT/mosestest2006utf.en \
    --destdir data-bin/europarl-en-moses \
    --workers 20