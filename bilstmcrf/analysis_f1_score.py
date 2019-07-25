#!/usr/bin/env python
#coding:utf-8
#author:zhujianqi

from __future__ import print_function
import sys
import traceback

def process(rfilename, wfilename):
    with open(rfilename, 'r') as f:
        line_num = 0
        info = []
        ele = []
        for line in f:
            line = line.strip('\r\n')
            if line.startswith('resotre'):
                line_num = 0
                continue
            try:
                index = line_num % 4
                if index == 0:
                    if ele and len(ele) == 11:
                        info.append(ele)
                    ele = []
                    #ele.append(int(line.split(',')[0].split(':')[1]))
                    ele.append(line.split(',')[0].split(':')[1])
                    ele.append(line.split(',')[1].split(':')[1])
                else:
                    w = line.split('\t')
                    #print(w[2] + '---' + w[3] + '---' + w[4])
                    for i in range(3):
                        #ele.append(float(w[i+2].split(' ')[1]))
                        ele.append(w[i+2].split(' ')[1])
            except Exception as e:
                print(str(line_num) + ":" + line)
                print(traceback.format_exc())
            line_num += 1
                # for test
                #if line_num == 5:
                #    break
        if ele and len(ele) == 11:
            info.append(ele)
    with open(wfilename, 'w') as f:
        for ele in info:
            f.write('{}\n'.format(' '.join(ele)))

def main():
    raw_filename = '../bin/bilstmcrf_pretrained/nohup.out.new'
    f1_filename = '../bin/bilstmcrf_pretrained/epoch_score.new'
    process(raw_filename, f1_filename)

if __name__ == "__main__":
    main()

