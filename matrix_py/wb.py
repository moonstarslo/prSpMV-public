from stringprep import b1_set
import scipy
import numpy as np

def mul_normal(mr,n):
    temp_x=[0.01 for j in range(mr.shape[0])]
    temp_vec=np.array([1 for i in range(mr.shape[0])])
    for time in range(n):
        for i in range(mr.shape[0]):
            temp_f=0
            for j in range(mr.indptr[i],mr.indptr[i+1]):
                # print(mr.data[mr.indices[j]],temp_x[mr.indices[j]])
                # print(temp_f,mr.data[mr.indices[j]],temp_x[mr.indices[j]],temp_x)
                temp_f=temp_f+mr.data[mr.indices[j]]*temp_x[mr.indices[j]]
            temp_vec[i]=temp_f
        for i in range(mr.shape[0]):
            temp_x[i]=temp_vec[i]
            #直接赋值-->同时改变
    #     print(temp_x)
    # print(temp_vec)
    return temp_x


#writeback without multi-threads
def writeback_single_threads(smr,seq,temp_x,a):
    times=0
    for colidx in seq.values():
        temp_x[times]=a[colidx]
        times=times+1
    return temp_x

def mul_spmm(smr,seq,n):
    a=np.array([1 for i in range(smr.shape[0])])
    #翻转key value方便找data位置
    seq_rotate={value:key for key,value in seq.items()}
    temp_x=[0.01 for j in range(len(seq))]
    for time in range(n):
        for i in range(smr.shape[0]):
            temp_f=0
            for j in range(smr.indptr[i],smr.indptr[i+1]):
                # print('time',time,temp_vec[smr.indices[j]],smr.data[seq_rotate[smr.indices[j]]])
                temp_f=temp_f+temp_x[smr.indices[j]]*smr.data[seq_rotate[smr.indices[j]]]#用temp_vec进行计算，应初始化
            a[i]=temp_f
        temp_x=writeback_single_threads(smr,seq_rotate,temp_x,a)
        #print(time)
    # print(temp_vec)
    # print(a)
    return a

def mul_spmm_bserial(smr,bseq,bcolidx,n):
    a=np.array([1 for i in range(smr.shape[0])])
    #翻转key value方便找data位置
    temp_x=[0.01 for j in range(len(bseq))]
    for time in range(n):
        for i in range(smr.shape[0]):
            temp_f=0
            for j in range(smr.indptr[i],smr.indptr[i+1]):
                #print('time',time,temp_vec[smr.indices[j]],smr.data[seq_rotate[smr.indices[j]]])
                temp_f=temp_f+temp_x[bcolidx[j]]*smr.data[bseq[bcolidx[j]]]#用temp_vec进行计算，应初始化
            a[i]=temp_f
        temp_x=writeback_single_threads(smr,bseq,temp_x,a)
        #print(time)
    # print(temp_vec)
    # print(a)
    return a

def list_to_dict(bseq):
    ind=0
    bseq_dict={}
    for i in range(len(bseq)):
        for colidx in bseq[i].keys():
            bseq_dict[ind]=colidx
            ind=ind+1
    return bseq_dict
#拟定数据结构：在seq内加入datavec记录temp_vec的值