import os
import argparse
import subprocess
import re


parser = argparse.ArgumentParser(description='Test kocr')
parser.add_argument('target_dir', type=str)
parser.add_argument('--kocr_cmd', type=str, default='../src/kocr ../databases/cnn-num.bin')
args = parser.parse_args()

if not args.target_dir.endswith('/'):
    args.target_dir += '/'

ans_chars = set()
ans_dict = {}
cnt_all = 0
cnt_suc = 0

for name in os.listdir(args.target_dir):
    extension = name.split('.')[-1]
    if extension not in ['png', 'jpg']:
        print 'ignored non image file:', name
        continue

    cmd = args.kocr_cmd + ' ' + args.target_dir + name
    res = subprocess.check_output(cmd, shell=True)
    mo = re.search('Result: (.)', res)
    
    if mo is None:
        print 'An error occured in testing', name
        continue

    pred = mo.group(1)
    ans = name[0]

    cnt_all += 1
    if pred == ans:
        cnt_suc += 1
    else:
        pass
        # subprocess.check_output('cp {} {}'.format(args.target_dir + name, '../miss/' + name[:-4] + '-' + pred + '.png'), shell=True)

    ans_chars.add(ans)
    if ans not in ans_dict:
        ans_dict[ans] = {}

    if pred in ans_dict[ans]:
        ans_dict[ans][pred] += 1
    else:
        ans_dict[ans][pred] = 1

ans_chars = sorted(list(ans_chars))
print '   ans |    ' + '    '.join(ans_chars)
print '-' * (8 + 5 * len(ans_chars))
for cr in ans_chars:
    print 'pred ' + cr + ' | ' + ' '.join(['{:4d}'.format(ans_dict[cc][cr] if cr in ans_dict[cc] else 0) for cc in ans_chars])

print 'Total accuracy:', float(cnt_suc) / cnt_all
