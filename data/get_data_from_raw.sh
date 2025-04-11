#!/bin/bash


python3 data/raw_train_test_sets.py "$@"
python3 data/combine_data.py "$@"
