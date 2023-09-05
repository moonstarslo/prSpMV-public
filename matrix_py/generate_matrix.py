#!/usr/bin/env python3

import os
import scipy
import scipy.io
import scipy.sparse


def output(output_path, M):
  data = list(M.data)
  colidx = list(M.indices)
  rowptr = list(M.indptr)
  rowb = rowptr[:-1]
  rowe = rowptr[1:]
  order = list(range(M.shape[0]))

  os.makedirs(output_path, exist_ok=True)

  with open(output_path + '/info.txt', 'w') as ff:
    ff.write(str(len(M.data)))
    ff.write('\n')
    ff.write(str(M.shape[0]))
    ff.write('\n')
    ff.write(str(M.shape[1]))
    ff.write('\n')
  with open(output_path + '/nnz.txt', 'w') as ff:
    for i in data:
      ff.write(str(i))
      ff.write(' ')
  with open(output_path + '/col.txt', 'w') as ff:
    for i in colidx:
      ff.write(str(i))
      ff.write(' ')
  with open(output_path + '/rowb.txt', 'w') as ff:
    for i in rowb:
      ff.write(str(i))
      ff.write(' ')
  with open(output_path + '/rowe.txt', 'w') as ff:
    for i in rowe:
      ff.write(str(i))
      ff.write(' ')


def main():
  matrix = scipy.io.mmread('as-caida.mtx')
  matrix = scipy.sparse.csr_matrix(matrix)
  output('data', matrix)


if __name__ == '__main__':
  main()
