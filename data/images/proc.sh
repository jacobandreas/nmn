#!/bin/bash

cc_models="/home/jda/3p/candc-1.00/models/questions"
cc_tok="/home/jda/3p/candc-1.00/bin/pos --model $cc_models/pos"
cc_parse="/home/jda/3p/candc-1.00/bin/parser --printer prolog --super $cc_models/super --model $cc_models/parser"

#echo "tokenizing"
#cat $1 | sed "s/\\?/ \\?/" | sed -r "s/ +/ /g" | sed -r "s/^ //" > $1.tok
#echo "tagging"
#cat $1.tok | $cc_tok > $1.pos
echo "parsing"
cat $1.pos | $cc_parse > $1.query
