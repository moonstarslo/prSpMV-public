import scipy
import scipy.io
import numpy as np
import os
from transmat import *

def get_row_lens(mr):
    rlen = []
    for r in range(mr.shape[0]):
        rlen.append(mr.indptr[r+1]-mr.indptr[r])
    return rlen

def get_same_row_lens(rlen):
    ret = [0] * 33
    for x in rlen:
        if x < 33:
            ret[x] += 1
    return ret

def gen_panel_list(mr, panel_num = 0):
    nnz = mr.indptr[-1]
    row = len(mr.indptr) - 1
    if panel_num == 0:
        panel_num = int(row / 2048) + 1
    if panel_num == 0:
        panel_num = 1
    panel_size = int(nnz / panel_num) + 1
    boundary = panel_size
    panel_list = [0]
    for i in range(0, row, 8):
        if mr.indptr[i] > boundary:
            boundary += panel_size
            panel_list.append(i)
    panel_list.append(row)
    # print("the correct number of panels is", panel_num)
    # print("the true number of panels is", len(panel_list) - 1)
    return panel_list

def panel_sort(mr, panelsize = 2 * 1024):
    print('Sort Panel Size {}'.format(panelsize))
    seq = get_row_lens(mr)
    pnum = int(len(seq)/panelsize)
    out_seq = []
    spv8_list = []
    iterations = 0
    if pnum == float(len(seq)) / float(panelsize):
        iterations = pnum
    else:
        iterations = pnum + 1
    for p in range(iterations):
        count = 0
        order = []
        remain = []
        same_len_rows = []
        spv8_len = 0
        if p == pnum:
            pseq = np.argsort(np.array(seq[p*panelsize:]))+p*panelsize
            same_len_rows = get_same_row_lens(seq[p*panelsize:])
        else:
            pseq = np.argsort(np.array(seq[p*panelsize:(p+1)*panelsize]))+p*panelsize
            same_len_rows = get_same_row_lens(seq[p*panelsize:(p+1)*panelsize])
        for i in range(33):
            begin = count
            end = count + same_len_rows[i]
            boundary = end - same_len_rows[i] % 8
            spv8_len += boundary - begin
            order.extend(pseq[begin:boundary])
            remain.extend(pseq[boundary:end])
            count += same_len_rows[i]
        spv8_list.append(spv8_len)
        remain.extend(pseq[count:])
        add_seq = order
        add_seq.extend(remain)
        out_seq.extend(list(add_seq))
    return out_seq, spv8_list

def panel_sort_nnz(mr, panelsize_list):
    seq = get_row_lens(mr)
    out_seq = []
    spv8_list = []
    for j in range(len(panelsize_list) - 1):
        count = 0
        order = []
        remain = []
        same_len_rows = []
        spv8_len = 0
        pseq = np.argsort(np.array(seq[panelsize_list[j]:panelsize_list[j+1]])) + panelsize_list[j]
        same_len_rows = get_same_row_lens(seq[panelsize_list[j]:panelsize_list[j+1]])
        for i in range(33):
            begin = count
            end = count + same_len_rows[i]
            boundary = end - same_len_rows[i] % 8
            spv8_len += boundary - begin
            order.extend(pseq[begin:boundary])
            remain.extend(pseq[boundary:end])
            count += same_len_rows[i]
        spv8_list.append(spv8_len)
        remain.extend(pseq[count:])
        add_seq = order
        add_seq.extend(remain)
        out_seq.extend(list(add_seq))
    return out_seq, spv8_list

def transpose_spv8(mr, spv8_list, panelsize = 2 * 1024):
    rowptr =  mr.indptr
    colidx = mr.indices
    data = mr.data
    new_rowptr = mr.indptr
    new_colidx = []
    new_data = []
    base = 0
    for i in range(len(spv8_list)):
        count = spv8_list[i]
        for row_index in range(base,base+count,8):
            rowlen = rowptr[row_index + 1] - rowptr[row_index]
            nnz_index = rowptr[row_index]
            add_data = [0] * rowlen * 8
            add_colidx = [0] * rowlen * 8
            for row in range(rowlen):
                for col in range(8):
                    add_data[8 * row + col] = data[nnz_index + col * rowlen + row]
                    add_colidx[8 * row + col] = colidx[nnz_index + col * rowlen + row]
            new_data.extend(add_data)
            new_colidx.extend(add_colidx)
        if i == (len(spv8_list) - 1):
            row_begin = base + count
            nnz_begin = rowptr[row_begin]
            new_data.extend(data[nnz_begin:])
            new_colidx.extend(colidx[nnz_begin:])
        else:
            row_begin = base + count
            row_end = base + panelsize
            nnz_begin = rowptr[row_begin]
            nnz_end = rowptr[row_end]
            new_data.extend(data[nnz_begin:nnz_end])
            new_colidx.extend(colidx[nnz_begin:nnz_end])
        base += panelsize
    return scipy.sparse.csr_matrix((new_data,new_colidx,new_rowptr),mr.shape)

def transpose_spv8_nnz(mr, spv8_list, panelsize_list):
    rowptr =  mr.indptr
    colidx = mr.indices
    data = mr.data
    new_rowptr = mr.indptr
    new_colidx = []
    new_data = []
    base = 0
    for i in range(len(spv8_list)):
        count = spv8_list[i]
        panelsize = panelsize_list[i+1] - panelsize_list[i]
        for row_index in range(base,base+count,8):
            rowlen = rowptr[row_index + 1] - rowptr[row_index]
            nnz_index = rowptr[row_index]
            add_data = [0] * rowlen * 8
            add_colidx = [0] * rowlen * 8
            for row in range(rowlen):
                for col in range(8):
                    add_data[8 * row + col] = data[nnz_index + col * rowlen + row]
                    add_colidx[8 * row + col] = colidx[nnz_index + col * rowlen + row]
            new_data.extend(add_data)
            new_colidx.extend(add_colidx)
        if i == (len(spv8_list) - 1):
            row_begin = base + count
            nnz_begin = rowptr[row_begin]
            new_data.extend(data[nnz_begin:])
            new_colidx.extend(colidx[nnz_begin:])
        else:
            row_begin = base + count
            row_end = base + panelsize
            nnz_begin = rowptr[row_begin]
            nnz_end = rowptr[row_end]
            new_data.extend(data[nnz_begin:nnz_end])
            new_colidx.extend(colidx[nnz_begin:nnz_end])
        base += panelsize
    return scipy.sparse.csr_matrix((new_data,new_colidx,new_rowptr),mr.shape)

def gen_spv8_list_txt(path,spv8_list):
    try:
        os.mkdir(path)
    except Exception as e:
        pass
    f = open('{}/spv8_list.txt'.format(path), 'w')
    for item in spv8_list:
        f.writelines(str(item)+' ')
    f.close()

def gen_spv8_panelsize_txt(path, panelsize_list):
    try:
        os.mkdir(path)
    except Exception as e:
        pass
    f = open('{}/spv8_panelsize.txt'.format(path), 'w')
    f.writelines(str(len(panelsize_list))+' ')
    for item in panelsize_list:
        f.writelines(str(item)+' ')
    f.close()

def gen_spv8_lists_txt(path,spv8_lists):
    try:
        os.mkdir(path)
    except Exception as e:
        pass
    count = 0
    f = open('{}/spv8_list_lens.txt'.format(path), 'w')
    f.writelines(str(len(spv8_lists))+' ')
    for li in spv8_lists:
        f.writelines(str(len(li))+' ')
        count += len(li)
    f.close()
    f = open('{}/spv8_list.txt'.format(path), 'w')
    f.writelines(str(count)+' ')
    for li in spv8_lists:
        for item in li:
            f.writelines(str(item)+' ')
    f.close()

def gen_sub_spv8_list_txt(path, spv8_list, index):
    f = open('{}/spv8_list_{}.txt'.format(path, index), 'w')
    count = len(spv8_list)
    f.writelines(str(count)+' ')
    for item in spv8_list:
        f.writelines(str(item)+' ')
    f.close()

#argsort排index
# x = [3,2,6,1,8,9,3,2,5,6,3,4,6,7,2,3,4,5,6,8,2,2,5,7,8,2,2,5,6,8]
# print(x)
# print(len(x))
# print(panel_sort(x, 4))
# print(len(panel_sort(x, 4)))

if __name__ == '__main__':
    data = list(range(1,21)) + list(range(1,7)) + list(range(1,9))
    data = data + list(range(1,21)) + list(range(1,10)) + list(range(1,4))
    rowptr = list(range(0,20,2)) + list(range(20,26,3)) + list(range(26,34))
    rowptr = rowptr + list(range(34,54,2)) + list(range(54,63)) + [63,66]
    colidx = [1] * 66
    mr = scipy.sparse.csr_matrix((data,colidx,rowptr),[40,40])
    seq_v8, spv8_list = panel_sort(mr,20)
    mr_v8 = reorder_row(mr, seq_v8)
    # print("seq_v8:")
    # print(seq_v8)
    # print("data: ")
    # print(mr_v8.data)
    # print("colidx: ")
    # print(mr_v8.indices)
    # print("rowptr: ")
    # print(mr_v8.indptr)
    # print("spv8_list:",spv8_list)
    ret = transpose_spv8(mr_v8, spv8_list,20)
    print("final data: ")
    print(ret.data)
    print("final colidx: ")
    print(ret.indices)
    print("final rowptr: ")
    print(ret.indptr)

