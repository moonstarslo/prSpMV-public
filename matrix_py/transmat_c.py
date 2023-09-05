import scipy
import scipy.io
import numpy as np
import os
from v8sort import *

def gen_mtx_txt(path,nnzcount,shape,nnz,colidx,rowptr,xsize):
    try:
        os.mkdir(path)
    except Exception as e:
        pass
    f = open('{}/info.txt'.format(path), 'w')
    f.writelines(str(nnzcount)+'\n')
    f.writelines(str(shape[0])+'\n')
    f.writelines(str(shape[1])+'\n')
    f.close()
    f = open('{}/col.txt'.format(path), 'w')
    for c in colidx:
        f.writelines(str(c)+' ')
    f.close()
    f = open('{}/row.txt'.format(path), 'w')
    for r in rowptr:
        f.writelines(str(r)+' ')
    f.close()
    f = open('{}/nnz.txt'.format(path), 'w')
    for n in nnz:
        f.writelines(str(n)+' ')
    f.close()
    f = open('{}/x.txt'.format(path), 'w')
    for x in range(xsize):
        f.writelines(str(1)+' ')
    f.close()

def reorder_row(mtx, seq):
    nnz = []
    colidx = []
    rowptr = [0]
    for s in seq:
        if mtx.indptr[s] == mtx.indptr[s+1]:
            rowptr.append(rowptr[-1])
        else:
            rowptr.append(rowptr[-1]+(mtx.indptr[s+1]-mtx.indptr[s]))
            for i in range(mtx.indptr[s], mtx.indptr[s+1]):
                nnz.append(mtx.data[i])
                colidx.append(mtx.indices[i])
    return scipy.sparse.csr_matrix((nnz, colidx, rowptr), shape=(mtx.shape))

def gen_panels(mtx, psize=256):
    rnum = mtx.shape[0]
    pnum = int(rnum/psize)
    plist = []
    for p in range(pnum+1):
        row_start = p*psize
        if p == pnum:
            row_end = rnum+1
        else:
            row_end = row_start + psize + 1
        # print(pnum,p,row_start,row_end,psize)
        p_indptr = mtx.indptr[row_start:row_end]
        p_nnz = mtx.data[p_indptr[0]:p_indptr[-1]]
        p_indices = mtx.indices[p_indptr[0]:p_indptr[-1]]
        offset = p_indptr[0]
        p_indptr = p_indptr - offset
        pm = scipy.sparse.csr_matrix((p_nnz, p_indices, p_indptr), shape=(row_end-row_start-1, mtx.shape[1]))
        plist.append(pm)
    return plist

def squeeze_row(mr):
    empty = []
    non_empty = []
    for r in range(mr.shape[0]):
        if mr.indptr[r+1]-mr.indptr[r] == 0:
            empty.append(r)
        else:
            non_empty.append(r)
    # print(len(non_empty),len(empty))
    nzsize = len(non_empty)
    non_empty.extend(empty)
    # print(len(non_empty),len(empty))
    mr = reorder_row(mr, non_empty)
    mrnz = gen_panels(mr, nzsize)
    # print(len(mrnz), mrnz[0].shape, mrnz[1].shape)
    return mrnz[0]

def main():
    mname = 'sx-askubuntu'
    # mname = 'web-Google'
    mpath = 'mat/mtx/{}/{}.mtx'.format(mname, mname)
    mtx = scipy.io.mmread(mpath)
    mr = scipy.sparse.csr_matrix(mtx)
    print('\n{}'.format(mname), mr.shape, mr.count_nonzero())
    mrnz = squeeze_row(mr)
    print('Non-Empty Rows: {}'.format(mrnz.shape[0]))
    print('\n{}'.format(mname), mrnz.shape, mrnz.count_nonzero())
    scipy.io.mmwrite('trace/tmp/{}-squeeze.mtx'.format(mname), mrnz)
    gen_mtx_txt('trace/tmp/{}-squeeze'.format(mname),mrnz.count_nonzero(),mrnz.shape,mrnz.data,mrnz.indices,mrnz.indptr,mrnz.shape[1])

    seq = panel_sort(get_row_lens(mrnz),1024*2)
    mrst = reorder_row(mrnz, seq)
    scipy.io.mmwrite('trace/tmp/{}-squeeze-sort.mtx'.format(mname), mrst)
    gen_mtx_txt('trace/tmp/{}-squeeze-sort'.format(mname),mrst.count_nonzero(),mrst.shape,mrst.data,mrst.indices,mrst.indptr,mrst.shape[1])

if __name__ == '__main__':
    main()