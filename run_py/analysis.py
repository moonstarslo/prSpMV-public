import os
from xml.dom.expatbuilder import parseFragmentString
import pandas as pd

def main():

    matrix_file = open("matrix.txt", "r")
    mlist = matrix_file.readlines()
    # mlist = ["com-LiveJournal.tar.gz ", "com-Orkut.tar.gz ", "soc-LiveJournal1.tar.gz "]
    bnum_list = [1, 4, 8, 16, 24, 32]
    # bnum_list.extend([40, 48, 56, 64, 96, 128])
    # bnum_list = [256, 512, 768, 1024, 1024+256, 1024+512, 1024+768]
    matrix_dict = {}

    for m in mlist:
        mname = m[:-8]
        # ./spmv 0 ../spmm_py/trace/{}/bitmap/v8/serial/{}-bserial-{} 1000 1 1 1
        # for bnum in bnum_list:
        info_file = open("../spmm_py/trace/{}/bitmap/v8/serial/{}-bserial-1/info.txt".format(mname,mname),"r")
        info_content =  list(info_file.readlines())
        nnz_size = info_content[0]
        nnz_size = int(nnz_size[:-1])
        row_size = info_content[1]
        row_size = int(row_size[:-1])
        matrix_dict[mname] = [nnz_size, row_size]
        for bnum in bnum_list:
            info_file = open("../spmm_py/trace/{}/bitmap/v8/serial/{}-bserial-{}/info.txt".format(mname,mname,bnum),"r")
            info_content =  list(info_file.readlines())
            seq_size = info_content[3]
            seq_size = int(seq_size[:-1])
            matrix_dict[mname].append(seq_size)
    index_list = ["nnz_size", "row_size"]
    for bnum in bnum_list:
        index_list.append("seq_size({} blocks)".format(bnum))
    res = pd.DataFrame(matrix_dict,index=index_list)
    res.to_csv("analysis_matrix.csv")


if __name__ == '__main__':
    main()
