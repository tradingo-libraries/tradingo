#!/usr/bin/env bash

export AIRFLOW_HOME=$PWD
export AIRFLOW__CORE__LOAD_EXAMPLES=false

tmux new-session -d -s airflow airflow standalone
tmux new-session -d -s dtale dtale --arcticdb-uri "lmdb://$PWD/test/arctic.db" --arcticdb-use_store
tmux new-session -d -s jupyter jupyter notebook -d ./notebooks/ --ip 0.0.0.0
