#!/bin/bash


#perl -pe"BEGIN {binmode STDIN,  ':encoding(cp1252)'; binmode STDOUT, ':encoding(UTF-8)';}" < dev2006.de > dev2006utf.de
#perl -pe"BEGIN {binmode STDIN,  ':encoding(cp1252)'; binmode STDOUT, ':encoding(UTF-8)';}" < devtest2006.de > devtest2006utf.de
#perl -pe"BEGIN {binmode STDIN,  ':encoding(cp1252)'; binmode STDOUT, ':encoding(UTF-8)';}" < europarl.de-en.de > europarlutf.de-en.de


SCRIPTS=mosesdecoder/scripts
NORM_PUNC=$SCRIPTS/tokenizer/normalize-punctuation.perl
REM_NON_PRINT_CHAR=$SCRIPTS/tokenizer/remove-non-printing-char.perl
TOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl
FOLDER=~/Desktop/europarl/
FILE=dev2006utf.de
INPUT_FILE=$FOLDER/$FILE
OUTPUT_FILE=$FOLDER/moses$FILE

l=de

cat $INPUT_FILE | \
    perl $NORM_PUNC $l | \
    perl $REM_NON_PRINT_CHAR | \
    perl $TOKENIZER -threads 8 -a -l $l >> $OUTPUT_FILE