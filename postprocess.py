'''
Date: 2020-10-22 22:34:50
LastEditors: Xi Chen(chenxi50@lenovo.com)
LastEditTime: 2020-10-23 01:18:09
'''
import yaml
import codecs
import re
import os

from glob import glob
from argparse import ArgumentParser
from attrdict import AttrDict

from tools.avg_checkpoints import avg_model
from tools.CER_Compute import compute_cer

re_sig = re.compile(r'\*')





if __name__ == "__main__":
  parser = ArgumentParser()
  parser.add_argument('-c', '--config', dest='config_file')
  args = parser.parse_args()

  f = open(args.config_file)
  config = yaml.load(f)
  config = AttrDict(config)

  os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
  
  print('step 1: load the label')
  ref_file = './data/dev_text'

  ref_dict = {}
  for line in codecs.open(ref_file, 'r', 'utf-8').readlines():
    line = line.strip()
    uttid,text = line.split(' ')
    ref_dict[uttid] = text

  
  print('step 2: read the dev results')
  dev_result_dir = config.dev.output_file+'*'
  cer_dict = {}
  for f in glob(dev_result_dir):
      # import pdb;pdb.set_trace()
      ids = f.split('_')[-1]
      to_process = []
      for line in codecs.open(f, 'r', 'utf-8').readlines():
        line = line.strip()
        line = re_sig.sub('',line)
        if len(line.split('\t'))!=2:
          uttid = line.split('\t')[0]
          text = u''
        else:
          uttid,text = line.split('\t')
        to_process.append((ref_dict[uttid], text))
      
      cer_dict[ids] = compute_cer(to_process)*100
      print(ids,cer_dict[ids])
      
  cer_dict_sorted = sorted(cer_dict.items(), key=lambda x: x[1])
  
  print(cer_dict_sorted)
  best_5_cer = cer_dict_sorted[:5]
  pre_str = os.path.join(config.model_dir, 'model_epoch_')
  checkpoints_list = []
  for item in best_5_cer:
    k = item[0]
    checkpoints_list.append(pre_str+k)
  avg_model_path = avg_model(checkpoints_list)
  
  os.system('CUDA_VISIBLE_DEVICES=1,2 python evaluate.py -c %s -ch True'%(args.config_file))
  import pdb;pdb.set_trace()

  avg_result = config.test.set1.output_path = '/'.join(config.test.set1.output_path.split('/')[:-1])+'/averaged_result.txt'
  
  to_process = []
  for line in codecs.open(avg_result, 'r', 'utf-8').readlines():
    line = line.strip()
    line = re_sig.sub('',line)
    if len(line.split('\t'))!=2:
        uttid = line.split('\t')[0]
        text = u''
    else:
        uttid,text = line.split('\t')
    to_process.append((ref_dict[uttid], text))
    
  final_cer = compute_cer(to_process)*100
  print('average model cer is: %.4f'%(final_cer))

  

      
    
  
        
        
  

    