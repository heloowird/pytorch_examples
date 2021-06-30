import sys

def stat(filename):
    gt = []
    pd = []
    tc = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip('\r\n')
            if line.find('groudth]:') != -1:
                gt.append(line.split(': ')[2])
            if line.find('predict]:') != -1:
                pd.append(line.split(': ')[2])
            if line.find('timecost]:') != -1:
                tc.append(float(line.split(': ')[2].replace(' ', ''))*1000)

    assert(len(pd) == len(gt))
    if tc:assert(len(pd) == len(tc))

    right_cnt = sum(1 if g == p else 0 for g, p in zip(gt, pd))
    avg_timecost = 0
    if tc: avg_timecost = sum(tc)/len(tc)
    print('total sent cnt: {}, sent top1 acc: {:.2f}%, avg timecost: {:.4f}'.format(len(gt), 100.0*right_cnt/len(gt), avg_timecost))


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: python {} predict_result_file'.format(sys.argv[0]))
        sys.exit(1)
    stat(sys.argv[1])

