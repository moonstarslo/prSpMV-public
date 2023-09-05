import os
import pandas as pd
import re

matrix_dict = {}
perf_dict = {}

def change_blocks(block_num):
    data = ''
    
    with open("spmv_csr_normal_verify.cpp","r+") as f:
        for line in f.readlines():
            if(line.find("#define BLOCK_NUM ") == 0):
                line = "#define BLOCK_NUM " + str(block_num) + "\n"
            data += line

    with open("spmv_csr_normal_verify.cpp","r+") as f:
        f.writelines(data)

    res = os.popen("make")
    # print(res)

def run_spmv_verify(mname):
    spmv_command = "numactl -C 12-23 --membind=1 "
    # spmv_command = "taskset -c 13,14,15,16 "
    # spmv_command = spmv_command + "../src/spmv_verify 0 ../matrix_py/trace_test/{}/v8_sort/{}-bserial-1 100".format(mname,mname)
    spmv_command = spmv_command + "../src/spmv_verify 0 ../matrix_py/trace/{}/{} 1000".format(mname,mname)
    res = os.popen(spmv_command)
    output_str_list = res.readlines()
    running_time = 0
    for output_str in output_str_list:
        if "average time:" in output_str:
            running_time = float(output_str[13:-1])
            print(mname,"matrix(pure)","running time: ",running_time)
            break
    if mname not in matrix_dict:
        matrix_dict[mname] = [running_time]
    else:
        matrix_dict[mname].append(running_time)

def run_spmv_verify_block(mname, block_num):
    spmv_command = "numactl -C 12-23 --membind=1 "
    # spmv_command = "taskset -c 12 "
    spmv_command = spmv_command + "../src/spmv_verify 0 ../matrix_py/trace_test/{}/{}-bserial-{} 100 1".format(mname,mname,block_num)
    res = os.popen(spmv_command)
    output_str_list = res.readlines()
    running_time = 0
    for output_str in output_str_list:
        if "average time:" in output_str:
            running_time = float(output_str[13:-1])
            print(mname,"matrix(pure)","running time: ",running_time)
            break
    if mname not in matrix_dict:
        matrix_dict[mname] = [running_time]
    else:
        matrix_dict[mname].append(running_time)

def run_spmv_test(mname, bnum, corenum=1):
    spmv_command = "numactl -C 12-23 --membind=1 "
    # spmv_command = "taskset -c 13,14,15,16 "
    # spmv_command = spmv_command + "../src/spmv_test 0 ../matrix_py/trace/{}/bitmap/v8/serial/{}-bserial-{} 1000 1 1 1".format(mname,mname,bnum)
    spmv_command = spmv_command + "../src/spmv {} ../matrix_py/trace_test/{}/bitmap/v8/serial/{}-bserial-{} 1000 1".format(corenum,mname,mname,bnum)
    # spmv_command = spmv_command + "../src/spmv_verify 0 ../matrix_py/trace/{}/{} 1000".format(mname,mname)
    # spmv_command = spmv_command + "../src/spmv {} ../matrix_py/trace/{}/bitmap/v8/serial/{}-bserial-{} 1000".format(corenum,mname,mname,bnum)
    # spmv_command = spmv_command + "../src/spmv_test 0 ../matrix_py/trace_test/{}/bitmap/v8/serial/{}-bserial-{} 1000 1 1 1".format(mname,mname,bnum)
    res = os.popen(spmv_command)
    output_str_list = res.readlines()
    running_time = 0
    for output_str in output_str_list:
        if "average cost:" in output_str:
            running_time = float(output_str[13:-1])
            print(mname,"matrix({} blocks)".format(bnum),"running time: ",running_time)
            break
    if mname not in matrix_dict:
        matrix_dict[mname] = [running_time]
    else:
        matrix_dict[mname].append(running_time)

def perf_test(mname, bnum, test_events, corenum=1):
    perf_prefix = "perf stat -e " 
    for index in range(len(test_events)):
        if index == len(test_events) - 1:
            perf_prefix += test_events[index] + " "
        else:
            perf_prefix += test_events[index] + ","
    spmv_command = perf_prefix + "numactl -C 12-23 --membind=1 "
    password = "20000715sxx"
    redirection_command = " 2> perf_test.txt"
    # spmv_command = "taskset -c 13,14,15,16 "
    # spmv_command = spmv_command + "../src/spmv {} ../matrix_py/trace_test/{}/bitmap/v8/serial/{}-bserial-{} 1000 1".format(corenum,mname,mname,bnum)
    spmv_command = spmv_command + "../src/spmv_verify 0 ../matrix_py/trace/{}/{} 1000".format(mname,mname)
    # spmv_command = spmv_command + "../src/spmv {} ../matrix_py/trace/{}/bitmap/v8/serial/{}-bserial-{} 1000".format(corenum,mname,mname,bnum)
    spmv_command += redirection_command
    res = os.popen('echo %s | sudo -S %s' % (password, spmv_command))
    output_str_list = res.readlines()
    f = open("perf_test.txt","r")
    output_str_list += f.readlines()
    for output_str in output_str_list:
        for event in test_events:
            if event in output_str:
                tmp_str_list = output_str.split(event)
                res = re.findall("\d*", str(tmp_str_list[0]))
                res = int("".join(res))
                name = mname + "({} blocks)".format(bnum)
                if name not in perf_dict:
                    perf_dict[name] = [res]
                else:
                    perf_dict[name].append(res)
    print("perf test of {} completes".format(mname))

def main():
    matrix_file = open("matrix.txt", "r")
    mlist = matrix_file.readlines()
    # mlist = ['com-LiveJournal.tar.gz ']
    # mlist = ["com-LiveJournal.tar.gz ", "com-Orkut.tar.gz ", "soc-LiveJournal1.tar.gz "]
    bnum_list = [1, 4, 8, 16, 24, 32]
    bnum_list.extend([40, 48, 56, 64, 96, 128])
    bnum_list = [1, 4, 8, 16, 32]
    # bnum_list = [256, 512, 768, 1024, 1024+256, 1024+512, 1024+768]
    bnum_list = [1]
    

    for i, m in enumerate(mlist):
        mname = m[:-8]
        # ../src/spmv 0 ../matrix_py/trace/{}/bitmap/v8/serial/{}-bserial-{} 100 1 1 1
        run_spmv_verify(mname)

        for bnum in bnum_list:
            run_spmv_test(mname,bnum)

    index_list = ["pure"]
    for bnum in bnum_list:
        index_list.append(str(bnum))

    res = pd.DataFrame(matrix_dict,index=index_list)
    res.to_csv("running_time_mutilplethreads.csv")

def new_block_test():
    matrix_file = open("matrix.txt", "r")
    mlist = matrix_file.readlines()
    
    # bnum_list = [88, 477, 1967, 32, 18, 22, 515, 10, 308, 102, 190, 411]
    

    for i, m in enumerate(mlist):
        mname = m[:-8]
        # ../src/spmv 0 ../matrix_py/trace/{}/bitmap/v8/serial/{}-bserial-{} 100 1 1 1
        # bnum = bnum_list[i]
        f = open('../matrix_py/trace_test/{}/bitmap/v8/serial/gen_block_num.txt'.format(mname),'r')
        bnum = f.readlines()
        bnum = int(bnum[0])
        f.close()

        run_spmv_test(mname, bnum)
        # run_spmv_verify(mname)
        matrix_dict[mname].append(bnum)


    index_list = ["uncertain", "block_num"]

    res = pd.DataFrame(matrix_dict,index=index_list)
    res.to_csv("running_time_mutilplethreads.csv")

def csr_test():
    matrix_file = open("matrix.txt", "r")
    mlist = matrix_file.readlines()
    # mlist = ['com-LiveJournal.tar.gz ']
    # mlist = ["com-LiveJournal.tar.gz ", "com-Orkut.tar.gz ", "soc-LiveJournal1.tar.gz "]
    bnum_list = [1, 4, 8, 16, 24, 32]
    bnum_list.extend([40, 48, 56, 64, 96, 128])
    # bnum_list = [1, 4, 8, 16, 32]
    # bnum_list = [256, 512, 768, 1024, 1024+256, 1024+512, 1024+768]
    # bnum_list = [1]
    
    for bnum in bnum_list:
        change_blocks(bnum)
        for m in mlist:
            mname = m[:-8]
            run_spmv_verify_block(mname,bnum)

    index_list = []
    for bnum in bnum_list:
        index_list.append(str(bnum))

    res = pd.DataFrame(matrix_dict,index=index_list)
    res.to_csv("running_time_mutilplethreads.csv")

def run_all_matrix_perf_test():
    matrix_file = open("matrix.txt", "r")
    mlist = matrix_file.readlines()
    # mlist = ['as-Skitter.tar.gz ']

    test_events = ["cache-references","cache-misses","L1-dcache-loads","L1-dcache-load-misses",\
                   "L1-dcache-stores","l2_rqsts.miss","l2_rqsts.references","LLC-load-misses","LLC-loads","LLC-store-misses","LLC-stores",\
                   "l2_lines_out.useless_hwpf","l2_rqsts.all_pf","l2_rqsts.pf_hit"]

    for i, m in enumerate(mlist):
        mname = m[:-8]
        # ../src/spmv 0 ../matrix_py/trace/{}/bitmap/v8/serial/{}-bserial-{} 100 1 1 1
        # bnum = bnum_list[i]
        f = open('../matrix_py/trace_test/{}/bitmap/v8/serial/gen_block_num.txt'.format(mname),'r')
        bnum = f.readlines()
        bnum = int(bnum[0])
        f.close()

        perf_test(mname, bnum, test_events)


    index_list = test_events

    res = pd.DataFrame(perf_dict,index=index_list)
    res.to_csv("running_time_mutilplethreads.csv")

if __name__ == '__main__':
    new_block_test()
    # run_all_matrix_perf_test()
