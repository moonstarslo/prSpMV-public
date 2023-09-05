#!/usr/bin/env python3

import os
from optparse import OptionParser
from cachesim import CacheSimulator,Cache,MainMemory

# parser = OptionParser()
# parser.add_option("-n","--matrixname",type="string",default="false")
# parser.add_option("-f","--traceformat",type="string",default="false")

# (options,args) = parser.parse_args()
# matrix_name = options.matrixname
# trace_format = options.traceformat

# thispath = os.path.dirname(os.path.realpath(__file__))
# data_path = os.path.join(thispath,'trace/',matrix_name)
# print(data_path)

def run_cachesim(trace_file):
    ## build memory system
    mem = MainMemory()
    # l2 = Cache("L2",128,16,64,"LRU")
    # l2 = Cache("L2",192,16,64,"LRU")
    l2 = Cache("L2",256,16,64,"LRU")
    # l2 = Cache("L2",1024,16,64,"LRU")
    mem.load_to(l2)
    mem.store_from(l2)
    l1 = Cache("L1",64,8,64,"LRU",load_from=l2,store_to=l2)
    cs = CacheSimulator(l1,mem)

    ## load trace from file
    trace = open(trace_file)
    trace = list(trace.readlines())
    trace_index = 0

    while trace_index < len(trace):
        addr = int(trace[trace_index],16)
        cs.load(addr,length=8)
        # cs.store(addr,length=8)
        trace_index += 1
        # addr_hex = hex(addr)
        # print(addr_hex)

    cs.force_write_back()
    print("matrix name:%s\n"%trace_file)
    cs.print_stats()
    print("\n")

if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("-n","--matrixname",type="string",default="false")
    parser.add_option("-f","--traceformat",type="string",default="false")
    (options,args) = parser.parse_args()
    matrix_name = options.matrixname
    trace_format = options.traceformat
    thispath = os.path.dirname(os.path.realpath(__file__))
    data_path = os.path.join(thispath,'trace/',matrix_name)
    trace_file = os.path.join(data_path,'{}.txt'.format(trace_format))
    print(data_path)
    run_cachesim(trace_file)
