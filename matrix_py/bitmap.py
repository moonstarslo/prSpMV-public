from cgitb import reset
from itertools import count
import scipy
import scipy.io
import numpy as np
import math
from transmat import *
from matrix_visualization import *

def get_scoreboard(mr,sectsize,thres=0):
    board = []
    #向上取整
    sectnum = math.ceil(mr.shape[1]/sectsize)
    print('Bitmap Sections={}'.format(sectnum))
    for i in range(mr.shape[0]):
        score = np.zeros((sectnum))
        if mr.indptr[i] == mr.indptr[i+1]:
            board.append(score)
            continue
        for n in mr.indices[mr.indptr[i]:mr.indptr[i+1]]:
            score[int(n/sectsize)] = score[int(n/sectsize)] + 1
        board.append(score)
    # print(board)
    return board

def gray_to_binary(num):
    print("num:", num) 
    x = bin(num)
    g = int(x[2])
    res = g
    for index in range(3, len(x)):
        g = (g + int(x[index])) % 2
        res = 2 * res + g
    return res

def blur(score):
    if max(score):
        #print(np.argmax(score)+1)
        return np.argmax(score)+1
    else:
        return 0
    # num = 10
    # res = 0
    # for i in range(num):
    #     if max(score):
    #         index = np.argmax(score)
    #         res = 2 ** index + res
    #         score[index] = 0
    #         print("res:", res)
    #     else:
    #         break
    # return gray_to_binary(res)

def gen_bitorder(scores):
    #map是python内置函数，会根据提供的函数对指定的序列做映射。
    bitseq = np.array(list(map(blur, scores)))
    #print(bitseq)
    #argsort返回从小到大的索引值
    return np.argsort(bitseq, kind='mergesort')

def bitmap_reorder(mr, sectsize):
    scores = get_scoreboard(mr, sectsize)
    row_seq = gen_bitorder(scores)
    #print(row_seq)
    return row_seq

if __name__ == '__main__':
    mname = "as-Skitter"
    mpath = 'mat/mtx/{}/{}.mtx'.format(mname, mname)
    mtx = scipy.io.mmread(mpath)
    mr = scipy.sparse.csr_matrix(mtx)
    img = gen_img(mr,1000,1000,normalized=False,all_or_none=False,fix_scale=True,dark=False)
    img.save('{}_origin_fixscale.png'.format(mname))
    row = len(mr.indptr) - 1
    sect = row / 32
    sect = 8 * 1024
    seq_bitmap = bitmap_reorder(mr, sect)
    mr = reorder_row(mr, seq_bitmap)
    img = gen_img(mr,1000,1000,normalized=False,all_or_none=False,fix_scale=True,dark=False)
    img.save('{}_bitmap_new_fixscale.png'.format(mname))


# mname = 'sx-superuser'
# mpath = 'mat/mtx/{}/{}.mtx'.format(mname, mname)
# mtx = scipy.io.mmread(mpath)
# mr = scipy.sparse.csr_matrix(mtx)
