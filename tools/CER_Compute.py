# -*- coding: UTF-8 -*-
import editdistance
from argparse import ArgumentParser

def compute_cer(results):
    """
    Arguments:
        results (list): list of ground truth and
        predicted sequence pairs.
    Returns the CER for the full set.
    """
    distance = sum(editdistance.eval(label, pred)
                for label, pred in results)
    total = sum(len(label) for label, _ in results)
    return float(distance)/float(total)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('-ref', '--ref', dest='reference')
    parser.add_argument('-res', '--res', dest='result')
    args = parser.parse_args()

    f = open(args.reference, 'r')
    ref = f.readlines()
    f.close()

    f = open(args.result, 'r')
    res = f.readlines()
    f.close()

    ref_dict = {}
    for line in ref:
        line = line.strip()
        uttid,text = line.split(' ')
        ref_dict[uttid] = text
    
    to_process = []
    for line in res:
        line = line.strip()
        uttid,text = line.split('\t')
        to_process.append((ref_dict[uttid], text))


    cer = compute_cer(to_process)
    print(cer)