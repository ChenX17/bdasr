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
  '''
  BAC009S0743W0495 河南商报记者张郁摄
  BAC009S0744W0121 单套总价三千万元以上的项目成交量达到一百套
  BAC009S0744W0122 亚豪机构市场部总监郭毅指出
  '''

  ref_dict = {}
  for line in codecs.open(ref_file, 'r', 'utf-8').readlines():
    line = line.strip()
    uttid,text = line.split(' ')
    ref_dict[uttid] = text

  
  print('step 2: read the dev results')
  dev_result_dir = config.dev.output_file+'*'
  cer_dict = {}
  for f in glob(dev_result_dir):
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
      
  print('step 3: average five best epoch')     
  cer_dict_sorted = sorted(cer_dict.items(), key=lambda x: x[1])
  
  best_5_cer = cer_dict_sorted[:5]
  print(best_5_cer)
  pre_str = os.path.join(config.model_dir, 'model_epoch_')
  checkpoints_list = []
  for item in best_5_cer:
    k = item[0]
    checkpoints_list.append(pre_str+k)
  avg_model_path = avg_model(checkpoints_list)
  
  print('step 4: decode with the averaged model')
  os.system('CUDA_VISIBLE_DEVICES=0,1,2,3 python evaluate.py -c %s -ch True'%(args.config_file))

  avg_result = config.test.set1.output_path = '/'.join(config.test.set1.output_path.split('/')[:-1])+'/averaged_result.txt'
  
  print('compute cer')
  ref_file = './data/test_text'
  '''
  BAC009S0903W0475 关于中国嵩山少林寺方丈齐永信的举报风波尚未停歇
  BAC009S0903W0476 因准儿媳的举报跌下神坛
  BAC009S0903W0477 位于温州苍南龙港镇水门村的一个仓库发生火灾
  '''

  ref_dict = {}
  for line in codecs.open(ref_file, 'r', 'utf-8').readlines():
    line = line.strip()
    uttid,text = line.split(' ')
    ref_dict[uttid] = text
  
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

  

      
    
  
        
        
  

    