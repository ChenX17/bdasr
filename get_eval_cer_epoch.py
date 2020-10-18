import os
import argparse
import logging
from third_party.tensor2tensor.avg_checkpoints_func import avg_model
from tools.asr_eval_tool.src.cer_cal_cn_en import compute_cer


parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument("--decode_results_path", type=str, default="")
parser.add_argument("--model_path", type=str, default="")
args = parser.parse_args()
outfile_dir = '/datadisk/projects/bd-asr/tools/asr_eval_tool/src/debug'
outfile_path = os.path.join(outfile_dir, args.decode_results_path.split('/')[-1]+'.log')
all_decode_result = os.listdir(args.decode_results_path)

logger = logging.getLogger(__name__)
logger.setLevel(level = logging.INFO)
handler = logging.FileHandler(outfile_path)
handler.setLevel(logging.INFO)
logger.addHandler(handler)

start = 0
items = len(all_decode_result)
items = 70
outputs = []
step = 1

for i in range(items+1):
    full_path = os.path.join(args.decode_results_path,'decode_result_epoch_' + str(start+i*step))
    if os.path.isfile(full_path):
        logger.info('epoch_%s'%str(start + i*step))
        #print('epoch_%s'% str(start + i*step))
        compute_cer('/datadisk/datasets/aishell1/dev/text', full_path, '', '', logger=logger)

# outfile : cer of each epoch
f = open(outfile_path,'r')
outfile = f.readlines()
f.close()
cer_dict = {}
for line in outfile:
    if "epoch" in line:
        #step = line.strip().split('_')[-1]
        step = 'model_' + line.strip()
    if "CER" in line:
        cer = line.split(':')[1].split('%')[0]
        cer_dict[step] = float(cer)
a = sorted(cer_dict.items(), key=lambda x: x[1])
print(a)
#print(cer_dict)
lowest_cer_list = []
count = 0
num = 5
for k,v in a:
    step = int(k.split('_')[-1])
    if count < num and step<=70:
        count += 1
        lowest_cer_list.append(k)
    else:
        continue
print(','.join(lowest_cer_list))
lowest_cer_model = ','.join(lowest_cer_list)
print('avg model: model path %s, best cer models %s', lowest_cer_model, args.model_path)
avg_model(lowest_cer_model, args.model_path)
