# Transformer with Bidirectional Decoder for Speech Recognition
The source code of the paper, Transformer with Bidirectional Decoder for Speech Recognition
### How to start?

`pip install -r requirement.txt`

`CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py -c configs/bd_8_4_512units_spec.yaml`

After training, you can use the postprecess.py to average the 5 best epoches, and get the CER of the averaged model.

`python postprecess.py -c configs/bd_8_4_512units_spec.yaml`
