#!/bin/bash
# This script start the TransformerASR job [train\eval\inference].
set -x
export LD_LIBRARY_PATH=/mnt/cephfs_new_wj/lab_speech/home/chenxi/codes/cudnn/cuda/lib64:$LD_LIBRARY_PATH
THIS_DIR="$( cd "$( dirname "$0" )" && pwd )"
EEASR_ROOT="$THIS_DIR/.."
export PYTHONPATH=$EEASR_ROOT:$PYTHONPATH
cd $EEASR_ROOT

name="" # Not used yet.
run_command=''
parse_commandline() {
    while [[ $# -gt 0 ]]; do
        item=$1
        key=$item
        case $key in
            --debug)
                DEBUG=true
                shift
                ;;
            --train)
                TRAIN=true
                shift
                ;;
            --eval)
                EVAL=true
                shift
                ;;
            --inference)
                INFERENCE=true
                shift
                ;;
            --name)
                shift
                NAME=$1
                shift
                ;;
            *)
                run_command="${run_command} ${key}"
                shift
                ;;
        esac
    done
}

parse_commandline "$@"

# Install some libs needed for TransformerASR.
pip install kaldi_io
pip install "/mnt/cephfs_hl/speech/home/nihao/package\
/tensorflow_gpu-1.11.0-cp27-cp27mu-manylinux1_x86_64.whl"


echo "run_command:" $run_command
if [[ $DEBUG ]]; then
    echo "Debug Begin..."
    echo "sleeping ..."
    sleep 2400h
    exit 0
fi

echo "**** Starting run.sh ..."

# For training.
if [[ $TRAIN ]]; then
    echo "Training Begin..."
    python eeasr/train_sb_asr.py $run_command
fi

# For eval.
if [[ $EVAL ]]; then
    # Not implement
    echo "Evaluating Begin..."
    python eeasr/evaluate_tokens_per_batch.py $run_command
fi

# For inference.
if [[ $INFERENCE ]]; then
    # Not implement
    echo "Inference with timestamp Begin..."
    python eeasr/evaluate_timestamp.py $run_command
fi

exit 0
