#!/usr/bin/env python3

import multiprocessing
import scipy
import scipy.io
import numpy as np
import os
import time
import functools
from transmat import *
#from cachemat import *
from bitmap import *
from v8sort import *
from wb import *
from wbsort import *
from bsize import *

origin = True

def gen_serial_origin(mr):
    seq_dict = {}
    seq_cnt = 0
    # 获得一个枚举，对每个元素，创建[index，data]
    # 获取col出现的index
    for i, col in enumerate(mr.indices):
        if col not in seq_dict.keys():
            seq_dict[col] = seq_cnt
            seq_cnt = seq_cnt + 1
            if seq_cnt == mr.shape[1]:
                # print('Have collected all col')
                break;
    new_seq = []
    for colidx in mr.indices:
        new_seq.append(seq_dict[colidx])
        #用新的colidx覆盖旧的(重排)
    new_mr = scipy.sparse.csr_matrix((mr.data, new_seq, mr.indptr), shape=mr.shape)
    return new_mr,seq_dict

def gen_serial(mr):
    seq_dict = {}
    seq_cnt = 0
    exist_colidx = np.zeros(mr.shape[1])
    for i, col in enumerate(mr.indices):
        exist_colidx[col] = 1
    for i in range(len(exist_colidx)):
        if exist_colidx[i] == 1:
            seq_dict[i] = seq_cnt
            seq_cnt = seq_cnt + 1
    new_seq = []
    for colidx in mr.indices:
        new_seq.append(seq_dict[colidx])
        #用新的colidx覆盖旧的(重排)
    new_mr = scipy.sparse.csr_matrix((mr.data, new_seq, mr.indptr), shape=mr.shape)
    return new_mr,seq_dict

def gen_trace_formats(mr, mname, nnz, bitmap=False, v8=False, v8_sort=False, serial=False):
    mpath = 'trace_test'

    try:
        os.mkdir(mpath)
    except Exception:
        pass

    mpath = 'trace_test/{}'.format(mname)

    try:
        os.mkdir(mpath)
    except Exception:
        pass

    gen_addr = True
    gen_mtx = True
    gen_wb = True    

    if bitmap:
        sect = 1024 * 8
        # sect = (len(mr.indptr) - 1) // 128
        # sect = 4 * 1024
        seq_bitmap = bitmap_reorder(mr,sect)
        mr = reorder_row(mr, seq_bitmap)
        mpath = mpath + '/bitmap'
        try:
            os.mkdir(mpath)
        except Exception as e:
            # print(e)
            pass
    else:
        seq_bitmap = np.array(list(range(len(mr.indptr)-1)))
    
    # panel_size = 2 * 1024
    
    if v8:
        mpath += '/v8'
        try:
            os.mkdir(mpath)
        except Exception:
            pass
    else:
        if v8_sort:
            mpath += '/v8_sort'
            try:
                os.mkdir(mpath)
            except Exception:
                pass

    if serial:
        mpath += '/serial'
        try:
            os.mkdir(mpath)
        except Exception:
            pass
    
    # block serialized
    #未替换mr内部的colidx，直接存在bserial_colidx内
    seq_v8_block=[]
    # print('****Block Serialized****')
    offset = 0
    bseq_list = []
    bserial_colidx = []
    bserial_data=[]
    bserial_indptr=[0]
    indptr_offset=0
    
    panel_size = 2048

    regions, bsize_list, bnum = gen_new_panels(mr)

    try:
        os.mkdir(mpath+'/{}-bserial-{}'.format(mname, bnum))
    except Exception:
        pass

    gen_bsize_txt(bsize_list, mpath+'/{}-bserial-{}'.format(mname, bnum) ,bnum)
    # print('Blocked Serial Number: {}'.format(bnum))
    cnt=0
    spv8_lists = []
    panelsize_list = []
    for i, r in enumerate(regions):
        if v8:
            add_panelsize_list = gen_panel_list(r)
            seq_v8, spv8_list = panel_sort_nnz(r, add_panelsize_list)
            tmp_r = reorder_row(r, seq_v8)
            r = transpose_spv8_nnz(tmp_r,spv8_list,add_panelsize_list)
            panelsize_list.extend(add_panelsize_list[:-1])
        else:
            if v8_sort:
                seq_v8, spv8_list = panel_sort(r, panel_size)
                r = reorder_row(r, seq_v8)
            else:
                spv8_list = []
                seq_v8 = np.array(list(range(len(r.indptr)-1)))

        for i in seq_v8:
            seq_v8_block.append(i+bsize_list[cnt])
        for i in r.data:
            bserial_data.append(i)
        for i in range(1,len(r.indptr)):
            bserial_indptr.append(r.indptr[i]+indptr_offset)
        indptr_offset=indptr_offset+r.indptr[len(r.indptr)-1]
        spv8_lists.append(spv8_list)
        cnt=cnt+1

        if serial:
            if origin:
                smr, seq = gen_serial_origin(r)
            else:
                smr, seq = gen_serial(r)
        else:
            smr = r
            seq = {}
            for i in range(r.shape[1]):
                seq[i] = i

        bseq_list.append(seq)
        
        for cc in smr.indices:
            bserial_colidx.append(cc+offset)
        offset = offset + len(seq.keys())

    # 剩下的都是将预处理生成txt文件的内容
    if gen_addr:
        bs_trace = open(mpath+'/bserial-{}.txt'.format(bnum), 'w')
        for p in bserial_colidx:
            bs_trace.writelines(hex(p*8)[2:]+'\n')
        bs_trace.close()
    if gen_mtx:
        gen_mtx_txt(mpath+'/{}-bserial-{}'.format(mname, bnum),nnz,mr.shape,bserial_data,bserial_colidx,bserial_indptr,mr.shape[1])
        if v8:
            gen_spv8_lists_txt(mpath+'/{}-bserial-{}'.format(mname, bnum),spv8_lists)
            gen_spv8_panelsize_txt(mpath+'/{}-bserial-{}'.format(mname, bnum),panelsize_list)
        SerialSort_block(seq_bitmap, seq_v8_block, bseq_list, path=mpath+'/{}-bserial-{}'.format(mname, bnum))
        f = open(mpath+'/{}-bserial-{}/rseq.txt'.format(mname, bnum), 'w')
        Rseq=SeqReverse(gen_rseq(seq_bitmap, seq_v8_block)).astype(int)
        for c in Rseq:
            f.writelines(str(c)+' ')
        f.close()
        f = open(mpath+'/{}-bserial-{}/sseq.txt'.format(mname, bnum), 'w')
        for c in bseq_list:
            for i in c.keys():
                f.writelines(str(i)+' ')
        f.close()

        f = open('{}/info.txt'.format(mpath+'/{}-bserial-{}'.format(mname, bnum)), 'a')
        f.writelines(str(bnum)+'\n')
        f.close()

        f = open(mpath+'/gen_block_num.txt', 'w')
        f.writelines(str(bnum)+' ')
        f.close()

    if gen_wb:
        wbseq1 = []
        wbseq2 = []
        for bseq in bseq_list:
            for k in bseq.keys():
                wbseq2.append(k)
        trace = open(mpath+'/bserial-{}-wb.txt'.format(bnum), 'w')
        for wb in wbseq2:
            trace.writelines(hex(wb*8)[2:]+'\n')
        trace.close()
        

def main():
    matrix_file = open("matrix.txt", "r")
    mlist = []
    for m in matrix_file.readlines():
        mlist.append(m)

    pro_list = []
    for m in mlist:
        mname = m[:-8]
        mpath = 'mat/mtx/{}/{}.mtx'.format(mname, mname)
        mtx = scipy.io.mmread(mpath)
        mr = scipy.sparse.csr_matrix(mtx)
        nnz = mr.count_nonzero()
        print('\n{}'.format(mname), mr.shape, mr.count_nonzero())

        pro = multiprocessing.Process(target = gen_trace_formats,args=(mr,mname,nnz,True,False,False,True))
        pro.start()
        
        pro_list.append(pro)
        print("the preprocess of {} starts".format(mname))

        if len(pro_list) > 10:
            for pro in pro_list:
                pro.join()
            pro_list.clear()

    for pro in pro_list:
        pro.join()
    print("all preprocess complete.")

if __name__ == '__main__':
    main()
