import numpy as np

def gen_seq_txt(seq, size,path=''):
    FileName ='seq'
    if path:
        path += '/'
    sseq = str(seq).replace(', ', ' ').replace('[', '').replace(']', '')
    f = open('{}.txt'.format(path + FileName), 'w')
    f.write(sseq)
    f.close()
    f = open('{}/info.txt'.format(path), 'a')
    f.writelines(str(size)+'\n')
    f.close()

def SeqReverse(seq):
    length = len(seq)
    reseq = np.zeros(length)
    for i in range(length):
        reseq[seq[i]] = i
    return reseq

def seqreverse_to_list(wseq_list, row_size):
    seq = []
    for l in wseq_list:
        seq.extend(l)
    reseq = [-1] * row_size
    for i in range(len(seq)):
        if reseq[seq[i]] == -1:
            reseq[seq[i]] = [i]
        else:
            reseq[seq[i]].append(i)
    return reseq

def gen_seqreverse_txt(seq, path = ''):
    FileName ='seqreverse'
    if path:
        path += '/'
    f1 = open('{}.txt'.format(path + FileName), 'w')
    f2 = open('{}.txt'.format(path + FileName + '_lens'), 'w')
    base = 0
    for l in seq:
        f2.writelines(str(base) + ' ')
        if l == -1:
            continue
        base += len(l)
        for item in l:
            f1.writelines(str(item) + ' ')
    f2.writelines(str(base) + ' ')
    f1.close()
    f2.close()

def gen_rseq(seq_bitmap, seq_v8):
    seq_row = seq_bitmap[seq_v8]#得到2次行重排最终的行重排顺序
    # print("seq_row =", seq_row)
    return seq_row

def gen_wseq(seq_row, seq_dict):
    seq_wb = []
    for key in seq_dict.keys():
        seq_wb.append(seq_row[key])
    # print(seq_wb)
    wseq = np.array(seq_wb)
    return wseq

def SerialSort(seq_bitmap, seq_v8, seq_dict, path=''):
    seq_row = gen_rseq(seq_bitmap, seq_v8)
    # print(seq_row)
    rseq=SeqReverse(seq_row)
    # print(rseq)
    seq_wb = list(gen_wseq(rseq, seq_dict).astype(int))
    gen_seq_txt(seq_wb,len(seq_wb),path)
    return seq_wb

def gen_wb_seqxy(mpath, wb_seq, row):
    f_x = open(mpath + "/seq_x.txt", "w")
    f_y = open(mpath + "/seq_y.txt", "w")
    wb_seq_list = []
    for item in wb_seq:
        wb_seq_list.extend(item)
    new_wb_dict = {}
    max_index = 0
    for index in range(len(wb_seq_list)):
        y = wb_seq_list[index]
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

def SerialSort_block(seq_bitmap, seq_v8, seq_list, path=''):
    wseq_list = []
    seq_row = gen_rseq(seq_bitmap, seq_v8)
    rseq=SeqReverse(seq_row)
    for seq in seq_list:
        wseq = list(gen_wseq(rseq, seq).astype(int))
        wseq_list.append(wseq)
    # len_seq=len(wseq_list)
    len_seq = 0
    for i in range(len(wseq_list)):
        len_seq=len_seq+len(wseq_list[i])
    gen_wb_seqxy(path, wseq_list, len(seq_bitmap))
    gen_seq_txt(wseq_list, len_seq, path)
    # gen_seqreverse_txt(seqreverse_to_list(wseq_list, len(seq_bitmap)), path)
    return wseq_list

# def main():
#     seq_bitmap = np.array([0,1,2,3])
#     seq_v8 = np.array([2,0,3,1])
#     seq_dict = {1:0,2:1,0:2,3:3}
#     seq_dict1 = {0:1, 1:2, 2:0, 3:4, 4:3}
#     seq_list = [seq_dict, seq_dict1]
#     seq_wb = SerialSort(seq_bitmap, seq_v8, seq_dict)
#     # wseq_list = SerialSort_block(seq_bitmap, seq_v8, seq_list)
#     print(seq_wb)
#     # print(wseq_list)

# main()
