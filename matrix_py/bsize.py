from re import L
import numpy as np
import scipy
import scipy.io
import os

def gen_bsize_txt(bsize_list, tfile,bnum):
    try:
        os.mkdir(tfile)
    except Exception as e:
        pass
    trace = open(tfile+'/bser.txt', 'w')
    for c in bsize_list:
        trace.writelines(str(c)+' ')
    trace.close()

def gen_bsize(indptr, bnum, bsize_offset, nnz_list):
    rnum = indptr.shape[0] - 1
    """ if rnum <= 8:
        bsize_list.append(rnum)
        for i in range(1,bnum):
            bsize_list.append(0)#当分块过多时，有些块大小为0（最后不足8行），加上这几行最终分块序列为[8,8,8,7,0]，去掉后生成[8,8,8,0,7]
    else: """
    if bnum == 1:
        bsize_offset.append(rnum)
        nnz_list.append(indptr[-1] - indptr[0])
    else:
        dnum = (indptr[-1] - indptr[0]) / bnum
        #print(dnum)
        bsize = 0
        for i in range(8, indptr.shape[0], 8):#以8行为一个单位，逐次累加，当块中元素数大于平均数时退出
            nnz = indptr[i] - indptr[0]
            if nnz > dnum:
                bsize = i
                nnz_list.append(nnz)
                break
        bsize_offset.append(bsize)
        gen_bsize(indptr[bsize:], bnum-1, bsize_offset, nnz_list)

def gen_bsize_list(mr, mpath, bnum=8):
    bsize_offset = []
    nnz_list = []
    gen_bsize(mr.indptr, bnum, bsize_offset, nnz_list)
    bsize_list = [0]
    for bsize in bsize_offset:
        bsize_list.append(bsize_list[-1] + bsize)
    bsize_list = list(map(int, bsize_list))
    nnz_list = list(map(int, nnz_list))
    gen_bsize_txt(bsize_list, mpath ,bnum)
    return bsize_list,nnz_list

def split_colidx(mr, bnum=8):
    mr_list = []
    block_size = mr.shape[1] // bnum
    for num in range(bnum):
        # print("block {} begins:".format(num))
        new_nnz = []
        new_rowptr = [0]
        count = 0
        new_colidx = []
        left_boundary = block_size * num
        right_boundary = left_boundary
        if num == (bnum - 1):
            right_boundary = mr.shape[1]
        else:
            right_boundary = left_boundary + block_size
        for row_index in range(len(mr.indptr) - 1):
            for nnz_index in range(mr.indptr[row_index],mr.indptr[row_index + 1]):
                if mr.indices[nnz_index] >= left_boundary and mr.indices[nnz_index] < right_boundary:
                    new_nnz.append(mr.data[nnz_index])
                    new_colidx.append(mr.indices[nnz_index] - left_boundary)
                    count += 1
            new_rowptr.append(count)
        tmp_mr = scipy.sparse.csr_matrix((new_nnz, new_colidx, new_rowptr), shape=(mr.shape[0],right_boundary - left_boundary))
        mr_list.append(tmp_mr)
    return mr_list
                
def split_matrix_col(mr, bnum):
    col = mr.shape[1]
    stride = int(col // bnum) + 1
    left_boundary = 0
    right_boundary = stride
    panels_list = []
    for index in range(bnum):
        data = []
        colidx = []
        rowptr = [0]
        counter = 0
        for i in range(len(mr.indptr) - 1):
            for j in range(mr.indptr[i], mr.indptr[i+1]):
                if mr.indices[j] >= left_boundary and mr.indices[j] < right_boundary:
                    data.append(mr.data[j])
                    colidx.append(mr.indices[j])
                    counter += 1
            rowptr.append(counter)
        tmp_mr = scipy.sparse.csr_matrix((data, colidx, rowptr), shape=mr.shape)
        panels_list.append(tmp_mr)
        left_boundary = right_boundary
        right_boundary += stride
    return panels_list

def main():
    #mtx = scipy.io.mmread('bcspwr01.mtx ')
    #mtx = scipy.io.mmread('sx-askubuntu.mtx ')
    mtx = scipy.io.mmread('./mat/mtx/com-LiveJournal/com-LiveJournal.mtx')
    mr = scipy.sparse.csr_matrix(mtx)
    mr_list = split_colidx(mr, 2)
    for sub_mr in mr_list:
        print("sub_mr:")
        print("nnz length:", len(sub_mr.data))
        print("sub_mr row length", sub_mr.shape[0])
        print("sub_mr col length", sub_mr.shape[1])

    #print(mr.indptr)
    # bsize_list, nnz_list = gen_bsize_list(mr, 'trace', 8)

if __name__ == '__main__':
    main()