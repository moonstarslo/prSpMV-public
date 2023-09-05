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

def gen_wb_seq(mpath, wb_seq):
    f_x = open(mpath + "/seq_x.txt", "w")
    f_y = open(mpath + "/seq_y.txt", "w")
    new_wb_dict = {}
    max_index = 0
    for index in range(len(wb_seq)):
        y = wb_seq[index]
        if y > max_index:
            max_index = y
        if y not in new_wb_dict:
            new_wb_dict[y] = [index]
        else:
            new_wb_dict[y].append(index)
    for index in range(max_index):
        if index in new_wb_dict:
            for x in new_wb_dict[index]:
                f_x.writelines(str(x) + ' ')
                f_y.writelines(str(index) + ' ')
    f_x.close()
    f_y.close()

def gen_colidx_trace(mr, tfile):
    trace = open(tfile, 'w')
    for c in mr.indices:
        trace.writelines(hex(c*8)[2:]+'\n')
    trace.close()

def add_colidx_trace(mr, mname, tname, offset):
    trace = open('trace/{}/{}.txt'.format(mname,tname), 'a')
    for c in mr.indices:
        trace.writelines(hex((c+offset)*8)[2:]+'\n')
    trace.close()

def cal_overlap(mr1, mr2):
    colidx1 = mr1.indices
    colidx2 = mr2.indices
    over = 0
    hist = []
    for c in colidx2:
        if c in colidx1 and c not in hist:
            over = over + 1
            hist.append(c)
    return over

def cnt_list(l):
    cnt = 0
    for i in range(len(l)-1):
        if l[i] == l[i+1]:
            cnt = cnt + 1
    return cnt

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
        # over = 0
        # if i > 0:
        #     over = cal_overlap(regions[i-1], r)

        if v8:
            # seq_v8, spv8_list = panel_sort(r, panel_size)
            # tmp_r = reorder_row(r, seq_v8)
            # r = transpose_spv8(tmp_r,spv8_list,panel_size)
            add_panelsize_list = gen_panel_list(r)
            # print("panelsize_list:", add_panelsize_list)
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

    if gen_addr:
        # print('Generate blocked serial trace')
        bs_trace = open(mpath+'/bserial-{}.txt'.format(bnum), 'w')
        for p in bserial_colidx:
            bs_trace.writelines(hex(p*8)[2:]+'\n')
        bs_trace.close()
    if gen_mtx:
        # print('Generate blocked serial mtx')
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
    # mlist = open('trace/list')
    # mlist = ['web-Stanford.tar.gz ', 'web-Google.tar.gz ', 'sx-superuser.tar.gz ']
    matrix_file = open("matrix.txt", "r")
    mlist = []
    for m in matrix_file.readlines():
        mlist.append(m)
    # mlist = ['caidaRouterLevel.tar.gz ','citationCiteseer.tar.gz ','coAuthorsCiteseer.tar.gz ']
    # mlist = ["com-LiveJournal.tar.gz ", "com-Orkut.tar.gz ", "soc-LiveJournal1.tar.gz "]
    # mlist = ['com-LiveJournal.tar.gz ']
    # mlist = ['web-Google.tar.gz ']
    # mlist.extend(['com-LiveJournal.tar.gz ','road-TX.tar.gz '])
    # mlist = ['bcspwr01.tar.gz ']
    # mlist=['neos.tar.gz ']
    bnum_list = [1, 4, 8, 16, 24, 32]
    # bnum_list.extend([40, 48, 56, 64, 96, 128])
    bnum_list = ['uncertain']
    # bnum_list = [256, 512, 768, 1024, 1024+256, 1024+512, 1024+768]
    for bnum in bnum_list:
        pro_list = []
        for m in mlist:
            mname = m[:-8]
            mpath = 'mat/mtx/{}/{}.mtx'.format(mname, mname)
            mtx = scipy.io.mmread(mpath)
            mr = scipy.sparse.csr_matrix(mtx)
            nnz = mr.count_nonzero()
            print('\n{}'.format(mname), mr.shape, mr.count_nonzero())

            # trace_path = 'trace/{}'.format(mname)
            # try:
            #     os.mkdir(trace_path)
            # except Exception as e:
            #     print(e)
            # bsize=8
            # bsize=1024*16

            # mr_list = split_colidx(mr, 2)
            # pro1 = multiprocessing.Process(target = gen_trace_formats,args=(mr_list[0],mname+"-left",nnz,bnum,True,True,False,True))
            # pro2 = multiprocessing.Process(target = gen_trace_formats,args=(mr_list[1],mname+"-right",nnz,bnum,True,True,False,True))
            # gen_trace_formats(mr, mname, nnz, bitmap=True, v8=True, v8_sort=False, serial=True)
            pro = multiprocessing.Process(target = gen_trace_formats,args=(mr,mname,nnz,True,False,True,True))
            pro.start()
            
            print("the preprocess of {} starts".format(mname))
        #     # pro1.start()
        #     # pro2.start()
        #     # pro_list.append(pro1)
        #     pro_list.append(pro)

        for pro in pro_list:
            pro.join()
            print("all preprocess complete.")

        # print("the preprocess of all matrix({}blocks) complete.".format(bnum))
        # run_cachesim(trace_path+'/load.txt')
        # run_cachesim(trace_path+'/serial.txt')
        # run_cachesim(trace_path+'/bserial-{}.txt'.format(bsize))
        # run_cachesim(trace_path+'/serial-wb.txt'.format(bsize))
        # run_cachesim(trace_path+'/bserial-{}-wb.txt'.format(bsize))

if __name__ == '__main__':
    main()
