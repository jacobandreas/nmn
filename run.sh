#!/bin/bash

export APOLLO_ROOT=/home/jda/3p/apollo
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$APOLLO_ROOT/build/lib
export PYTHONPATH=$PYTHONPATH:$APOLLO_ROOT/python:$APOLLO_ROOT/python/caffe/proto

#python -u -m cProfile main.py -c config/images_nmn.yml |& tee profile
#python main.py -c config/images_lstm.yml
#kernprof -l main.py -c config/cocoqa_nmn.yml
#python simple.py
#python bug.py

#python main.py -c config/cocoqa_ensemble.yml
#python main.py -c config/cocoqa_lstm.yml
python main.py -c config/cocoqa_nmn_alt.yml
#python main.py -c config/cocoqa_nmn.yml
