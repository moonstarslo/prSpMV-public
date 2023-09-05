#!/usr/bin/env python3

import scipy
import scipy.io
import numpy as np
import os
from transmat import *

def label_hotspot(mtx, hotlist):
    label = {}
    keys = hotlist.keys()
    for r in range(mtx.shape[0]):
        label[r] = 0
        if(mtx.indptr[r] != mtx.indptr[r]+1):
            for i in range(mtx.indptr[r], mtx.indptr[r+1]):
                if mtx.indices[i] in keys:
                    label[r] = label[r] + hotlist[mtx.indices[i]]
    return label

def gen_dense(mtx, threshold=8):
    mc = mtx.tocsc()
    clen = []
    for c in range(mc.shape[1]):
        clen.append(mc.indptr[c+1]-mc.indptr[c])
    nnz = 0
    col = []
    for ic, cc in enumerate(clen):
        if cc >= threshold:
            nnz = nnz+cc
            col.append(ic)
    return [nnz, col]

def aspt(mr, psize=256, threshold=8):
    plist = gen_panels(mr, psize)
    dense_nnz = 0
    dense_col = 0
    for p in plist:
        dsize = gen_dense(p, threshold)
        dense_nnz += dsize[0]
        dense_col += len(dsize[1])
    return [dense_nnz, dense_col]

def opt_mtx(mtx, mr, topk=16):
    # find hot spots
    mc = scipy.sparse.csc_matrix(mtx)
    clen_dict = {}
    for c in range(mc.shape[1]):
        l = mc.indptr[c+1]-mc.indptr[c]
        clen_dict[c] = l
    clen_sorted = sorted(clen_dict.items(), key=lambda x: x[1], reverse=True)
    hotlist = {}
    for i in range(topk):
        hotlist[clen_sorted[i][0]] = pow(2,i)

    # reorder matrix rows
    labeld_rows = label_hotspot(mr, hotlist)
    label_sorted = sorted(labeld_rows.items(), key=lambda x: x[1], reverse=True)
    row_seq = list(map(lambda x:x[0], label_sorted))
    reorder_mr = reorder_row(mr, row_seq)
    return reorder_mr
    
def region_aspt(mr, psize=256, threshold=8, topk=16, regionsize=256*40):
    regions = gen_panels(mr, regionsize)
    dense_nnz = 0
    dense_col = 0
    for r in regions:
        ropt = opt_mtx(r, r, topk)
        res = aspt(ropt, psize, threshold)
        dense_nnz = dense_nnz + res[0]
        dense_col = dense_col + res[1]
    return [dense_nnz, dense_col]

def main():

    # indptr  = np.array([0, 2, 4, 7])
    # indices = np.array([0, 2, 0, 2, 0, 1, 2])
    # data    = np.array([1, 2, 3, 4, 5, 6, 7])
    # a = scipy.sparse.csr_matrix((data, indices, indptr), shape=(3, 3))
    # print(a.toarray())
    # print(a.indices)
    # b = a.tocsc()
    # print(b.toarray())
    # print(b.indices)
    # plist = gen_panels(a, 2)
    # c = reorder_row(a, [2,0,1])
    # print(c.toarray())
    # print(label_hotspot(a, {0:1,1:2,2:4}))

    thres = 8
    topk = 32
    panel = 256
    region = 256*40
    print('threshold={} top-k={} panel={} region={}'.format(thres, topk, panel, region))
    mlist = open('mat/src/1/list')
    # mlist = ['as-caida.tar.gz ']
    for m in mlist:
        mname = m[:-8]
        mpath = 'mat/mtx/{}/{}.mtx'.format(mname, mname)
        print(mname)
        mtx = scipy.io.mmread(mpath) 
        mr = scipy.sparse.csr_matrix(mtx)
        print(mr.shape, mr.count_nonzero())
        if mr.shape[0] > 500000:
            print('SKIP')
            continue
        d = aspt(mr, threshold=thres, psize=panel)
        if d[0]:
            print("Dense NNZ={} Density={:.2f} Dense%={:.2f}".format(d[0], d[0]/d[1], d[0]/mr.count_nonzero()))
        else:
            print("Dense NNZ=0 Density=0 Dense%=0")
        # print("Dense NNZ=0 Density=0 Dense%=0")

        opt_mr = opt_mtx(mtx, mr, topk=topk)
        d2 = aspt(opt_mr, threshold=thres)
        if d2[0]:
            print("Dense NNZ={} Density={:.2f} Dense%={:.2f}".format(d2[0], d2[0]/d2[1], d2[0]/mr.count_nonzero()))
        else:
            print("Dense NNZ=0 Density=0 Dense%=0")
        
        d3 = region_aspt(mr, panel, thres, topk, region)
        if d3[0]:
            print("Dense NNZ={} Density={:.2f} Dense%={:.2f}".format(d3[0], d3[0]/d3[1], d3[0]/mr.count_nonzero()))
        else:
            print("Dense NNZ=0 Density=0 Dense%=0")
    mlist.close()


if __name__ == '__main__':
  main()
