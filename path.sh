# nihao's environment on 10.10.56.12
export PATH='/data00/home/nihao/software/anaconda2/bin':$PATH
#export PATH='/data00/home/nihao/software/anaconda3/bin':$PATH #python3 if need
export PATH='/data00/home/nihao/software/openmpi/bin':$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-9.0/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/cuda-10.0/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH='/data00/home/nihao/software/openmpi/lib':$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/data00/home/nihao/software/nccl_2.3.7-1+cuda9.0_x86_64/lib:$LD_LIBRARY_PATH

working_dir=`pwd`
export PYTHONPATH=$working_dir:$PYTHONPATH
