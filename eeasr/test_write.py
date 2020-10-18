import random
scp_files = ['/datadisk/softwares/kaldi/egs/aishell/s5/fbank/raw_fbank_train.1.scp','/datadisk/softwares/kaldi/egs/aishell/s5/fbank/raw_fbank_train.10.scp','/datadisk/softwares/kaldi/egs/aishell/s5/fbank/raw_fbank_train.5.scp']
total_scp = []
for index,scp_path in enumerate(scp_files):
      f = open(scp_path, 'r')
      scp = f.readlines()
      f.close()
      if index == 0:
        total_scp = scp
      else:
        total_scp += scp
        random.shuffle(total_scp)
f = open('test.scp','w')
f.writelines(total_scp)
f.close()