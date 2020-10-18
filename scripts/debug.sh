#!/bin/bash
# This script start the TransformerASR job [train\eval\inference].
set -x

THIS_DIR="$( cd "$( dirname "$0" )" && pwd )"
EEASR_ROOT="$THIS_DIR/.."
export PYTHONPATH=$EEASR_ROOT:$PYTHONPATH
cd $EEASR_ROOT

STAGE=1

if [[ $STAGE == 1 ]]; then
  run_command='-d'
  python eeasr/train.py $run_command
fi

