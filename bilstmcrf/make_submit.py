#!/usr/bin/env python
#coding:utf-8
#author:zhujianqi

from __future__ import print_function
import sys

def make_submit(input_file, output_file):
    with open(output_file, 'w') as wf:
        with open(input_file) as rf:
            output = ''
            last_lb = ''
            last_chs = []
            for line in rf:
                line = line.strip('\r\n')

                if not line:
                    output += '%s/%s' % ('_'.join(last_chs), last_lb)
                    wf.write('%s\n' % (output.rstrip(' ')))

                    last_lb = ''
                    last_chs = []
                    output = ''
                    continue

                info = line.split()
                ch = info[0]
                lb = info[1].split('-')[-1]

                if not last_lb:
                    last_lb = lb

                if lb != last_lb:
                    if last_chs:
                        output += '%s/%s  ' % ('_'.join(last_chs), last_lb)
                    last_lb = lb
                    last_chs = []

                last_chs.append(ch)
            #output += '%s/%s ' % ('_'.join(last_chs), last_lb)
            #wf.write('%s\n' % (output.rstrip(' ')))

def main():
    input_filename = sys.argv[1]
    output_filename = sys.argv[2]
    make_submit(input_filename, output_filename)

if __name__ == "__main__":
    main()

