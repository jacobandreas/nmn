#!/bin/bash

export APOLLO_ROOT=/home/jda/3p/apollo
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$APOLLO_ROOT/build/lib
export PYTHONPATH=$PYTHONPATH:$APOLLO_ROOT/python:$APOLLO_ROOT/python/caffe/proto

#python -u -m cProfile main.py -c config/images_nmn.yml |& tee profile
#kernprof -l main.py -c config/images_nmn.yml
python main.py -c config/images_nmn.yml
#python main.py -c config/images_lstm.yml
#kernprof -l bug.py
#python simple.py
