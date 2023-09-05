#include <immintrin.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <iostream>
#include <fstream>
#include <chrono>
#include <vector>
#include <math.h>
#include "omp.h"
#include <cassert>


using namespace std;
using namespace chrono;


// 默认线程数是1
int threadnum=1;

int row, col, nnz, seq_size, bnum;

// parameters of sparse matrix
double *val; // 稀疏矩阵CSR中的非零元元素数组
int *col_idx; // 稀疏矩阵CSR中的非零元元素对应列号的数组
int *row_ptr; // 稀疏矩阵CSR中的划分不同行的数组
int *vec_ptr; // 划分serial块的数组， 行号大于等于vec_ptr[0]小于vec_ptr[1]的为serial第一块

// parameters of spv8
int *spv8_list_lens; // 每个serial块中含有多少个V8 panel
int *spv8_lists; // 每个V8块中 cross-parallel处理的行数，也就是跨行进行SIMD的行数
int *spv8_panelsize; // 每个V8块的行数， V8块的行数不固定

// parameters of vector x
double *vec; // y = Ax中的x，注意此时的y不是正常CSR运算时的顺序，因为预处理中含有列重排（serial）
double *ret; // y = Ax中的y，注意此时的y不是正常CSR运算时的顺序，因为预处理中含有行重排（V8，bitmap）
double *ans; // 最终重排回正常顺序的y

// parameters of order
int *seq; // 记录下次迭代的x在本次迭代中y的位置的数组
int *rseq; // 记录行重拍顺序的数组
int *sseq; // 记录列重排顺序的数组

int run_num; // 迭代次数
const int warm_num = 300; // 预热次数

vector<vector<int> > begin_vec; // 记录每次要传给avx512_spvv8_kernel_tr函数传参rowptr位置的数组，begin_vec[0]记录的是第一个V8块中的内容，这样做可以减少kernel中对于中间变量的运算
vector<vector<int> > row_index_vec; // 记录每次要传给avx512_fma_spvv_kernel函数传参col,nnz位置的数组，row_index_vec[0]记录的是第一个V8块中的内容，这样做可以减少kernel中对于中间变量的运算

double time_wb=0;

void write_back()
{
    // 对写回进行多核调度时，每个核会在自己空闲的时候被赋予写回下一个1024行的任务
    static const int wb_block_size = 1024;
    omp_set_num_threads(threadnum);
    #pragma omp parallel for schedule(dynamic)
    for(int i = 0; i < seq_size; i += wb_block_size)
        for(int j = i; j < min(seq_size, i + wb_block_size); j++)
            vec[j]=ret[seq[j]];  
}

void generate_temperory_variables()
{
    int total_panels_num = 0;
    for(int i = 0; i < bnum; i++)   total_panels_num += spv8_list_lens[i];

    for(int k = 0; k < total_panels_num; k++)
    {
        int i = 0;
        int j = k;

        for(int index = 0; index < bnum; index++)
            if(j >= spv8_list_lens[index])
            {
                j = j - spv8_list_lens[index];
                i++;
            }
            else
                break;

        int row_begin = vec_ptr[i];
        int row_end = vec_ptr[i+1];
        int panels_num = spv8_list_lens[i];
        int spv8_len = spv8_lists[k];

        vector<int> add_begin_vec, add_row_index_vec;

        for(int row = 0; row < spv8_len; row += 8)
        {
            int begin = row_begin + spv8_panelsize[k] + row;
            int rowlen = row_ptr[begin + 1] - row_ptr[begin];
            if(rowlen != 0)
                add_begin_vec.push_back(begin);
        }
        if( j == (panels_num - 1))
        {
            int begin = row_begin + spv8_panelsize[k] + spv8_len;
            int end = row_end;
            for(int row_index = begin; row_index < end; row_index++)
            {
                int rowlen = row_ptr[row_index + 1] - row_ptr[row_index];
                if(rowlen != 0)
                    add_row_index_vec.push_back(row_index);
            }
        }
        else
        {
            int begin = row_begin + spv8_panelsize[k] + spv8_len;
            int end = row_begin + spv8_panelsize[k+1];
            for(int row_index = begin; row_index < end; row_index++)
            {
                int rowlen = row_ptr[row_index + 1] - row_ptr[row_index];
                if(rowlen != 0)
                    add_row_index_vec.push_back(row_index);
            }
        }
        
        begin_vec.push_back(add_begin_vec);
        row_index_vec.push_back(add_row_index_vec);

        j++;
    }
}

// in-row parallel， 在1行中每8个元素进行SIMD操作
double avx512_fma_spvv_kernel(int *col, double *nnz, int rowlen,
                                        double *x) {
  int limit = rowlen - 7;
  int *col_p;
  double *nnz_p;
  double sum = 0;
  __m256i c1;
  __m512d v1, v2, s;
  s = _mm512_setzero_pd();
  int i;
  
  for (i = 0; i < limit; i += 8) {
    col_p = col + i;
    nnz_p = nnz + i;
    c1 = _mm256_loadu_si256((const __m256i *) col_p);
    v2 = _mm512_i32gather_pd(c1, x, 8); // return {x[c1[0]],x[c1[1]]...x[c1[7]]}
    v1 = _mm512_loadu_pd(nnz_p);
    s = _mm512_fmadd_pd(v1, v2, s); // return v1 * v2 + s
  }

  sum += _mm512_reduce_add_pd(s);
  for (; i < rowlen; i++) {
    sum += nnz[i] * x[col[i]];
  }

  return sum;
}

// cross-row parallel， 相邻8行中第i个元素进行SIMD操作， 由于V8预处理中进行转置， 所以相邻8行的第i个元素是相邻的
void avx512_spvv8_kernel_tr(int *rowptr,
                            int *col, double *nnz, double *x,
                            double *y) {
    static const int rows[8] = {0,1,2,3,4,5,6,7};
    __m256i rs = _mm256_loadu_si256((const __m256i *) rows);
    __m512d acc = _mm512_setzero_pd();

    int rowlen = *(rowptr + 1) - *rowptr;
    int base = *rowptr;

  for (int c = 0; c < rowlen; c++) {
    int offset = base + c * 8;
    __m256i cc = _mm256_loadu_si256((const __m256i *) (col + offset));
    __m512d nz = _mm512_loadu_pd(nnz + offset);
    __m512d xx = _mm512_i32gather_pd(cc, x, 8);
    acc = _mm512_fmadd_pd(nz, xx, acc);
  }

  _mm512_i32scatter_pd(y, rs, acc, 8);
}

// 以V8块进行线程调度，每个线程块的行数不固定
void spmv_kernel()
{
    omp_set_num_threads(threadnum);
    #pragma omp parallel for schedule(dynamic)
    for(int i=0; i < begin_vec.size(); i++)
    {
        vector<int> & tmp_begin_vec = begin_vec[i];
        for(int j = 0;j < tmp_begin_vec.size(); j++)
            avx512_spvv8_kernel_tr(&row_ptr[tmp_begin_vec[j]],col_idx,val,vec,&ret[tmp_begin_vec[j]]);
        vector<int> & tmp_vec = row_index_vec[i];
        for(int j = 0;j < tmp_vec.size(); j++)
        {
            int row_index = tmp_vec[j];
            int rowlen = row_ptr[row_index + 1] - row_ptr[row_index];
            int nnz_begin = row_ptr[row_index];
            _mm_prefetch(ret + row_index, _MM_HINT_ET1);
            ret[row_index] = avx512_fma_spvv_kernel(&col_idx[nnz_begin],&val[nnz_begin],rowlen,vec);
        }
    }

    auto begin_wb = high_resolution_clock::now();
    write_back();
    auto end_wb = high_resolution_clock::now();
    auto duration_wb = duration_cast<microseconds>(end_wb - begin_wb);

    time_wb += double(duration_wb.count());
}

void handle_error(){
    cout<<"Error in loading data."<<endl;
    exit(0);
}

// 各参数的解释分别在他们的定义处
void read_data(string route){
    FILE *fp;
    string file_route;
    ifstream inFile;

    // read data from info.txt
    inFile.open(route + "/info.txt");
    if (!inFile) {
        cerr << "Unable to open file datafile.txt";
        exit(1);   // call system to stop
    }
    if (inFile) inFile>>nnz>>row>>col>>seq_size>>bnum;
    inFile.close();

    inFile.open(route + "/bser.txt");
    vec_ptr = new int [bnum + 1];
    if (!inFile) {
        cerr << "Unable to open file datafile.txt";
        exit(1);   // call system to stop
    }
    if (inFile) for (int i = 0;i < bnum + 1;i++) inFile >> vec_ptr[i];
    inFile.close();

    inFile.open(route + "/x.txt");
    vec = new double [max(seq_size,col)];
    if (!inFile) {
        cerr << "Unable to open file datafile.txt";
        exit(1);   // call system to stop
    }
    if (inFile) for (int i = 0;i < col;i++) inFile >> vec[i];
    inFile.close();

    inFile.open(route + "/row.txt");
    row_ptr = new int [row + 1];
    if (!inFile) {
        cerr << "Unable to open file datafile.txt";
        exit(1);   // call system to stop
    }
    if (inFile) for (int i = 0;i <= row;i++) inFile >> row_ptr[i];
    inFile.close();

    inFile.open(route + "/col.txt");
    col_idx = new int [nnz];
    if (!inFile) {
        cerr << "Unable to open file datafile.txt";
        exit(1);   // call system to stop
    }
    if (inFile) for (int i = 0;i < nnz;i++) inFile >> col_idx[i];
    inFile.close();

    inFile.open(route + "/nnz.txt");
    val = new double [nnz];
    if (!inFile) {
        cerr << "Unable to open file datafile.txt";
        exit(1);   // call system to stop
    }
    if (inFile) for (int i = 0;i < nnz;i++) inFile >> val[i];
    inFile.close();

    inFile.open(route + "/rseq.txt");
    rseq = new int [col];
    if (!inFile) {
        cerr << "Unable to open file datafile.txt";
        exit(1);   // call system to stop
    }
    if (inFile) for (int i = 0;i < col;i++) inFile >> rseq[i];
    inFile.close();

    inFile.open(route + "/sseq.txt");
    sseq = new int [seq_size];
    if (!inFile) {
        cerr << "Unable to open file datafile.txt";
        exit(1);   // call system to stop
    }
    if (inFile) for (int i = 0;i < seq_size;i++) inFile >> sseq[i];
    inFile.close();

// int spv8_list_lens[max_row];
// int spv8_lists[max_row];
    int count = 0;
    inFile.open(route + "/spv8_list_lens.txt");
    if (!inFile) {
        cerr << "Unable to open file spv8_list_lens.txt\n\r";
        // exit(1);   // call system to stop
    }
    inFile >> count;
    spv8_list_lens = new int [count];
    if (inFile) for (int i = 0;i < count;i++) inFile >> spv8_list_lens[i];
    if (inFile) inFile.close();

    // int count = 0;
    inFile.open(route + "/spv8_list.txt");
    if (!inFile) {
        cerr << "Unable to open file spv8_list.txt\n\r";
        // exit(1);   // call system to stop
    }
    inFile >> count;
    spv8_lists = new int [count];
    if (inFile) for (int i = 0;i < count;i++) inFile >> spv8_lists[i];
    if (inFile) inFile.close();

    inFile.open(route + "/spv8_panelsize.txt");
    if (!inFile) {
        cerr << "Unable to open file spv8_panelsize.txt\n\r";
        // exit(1);   // call system to stop
    }
    inFile >> count;
    spv8_panelsize = new int [count];
    if (inFile) for (int i = 0;i < count;i++) inFile >> spv8_panelsize[i];
    if (inFile) inFile.close();

    inFile.open(route + "/seq.txt");
    seq = new int [seq_size];
    if (!inFile) {
        cerr << "Unable to open file datafile.txt";
        exit(1);   // call system to stop
    }
    if (inFile) for (int i = 0;i < seq_size;i++) inFile >> seq[i];
    inFile.close();

    ret = new double [max(seq_size,col)];
    ans = new double [row];

    // for (int i = 0;i<bnum+1;i++) cout<<vec_ptr[i]<<" ";
    for(int i=0;i<seq_size;i++) ret[i]=vec[sseq[i]];
    for(int i=0;i<seq_size;i++) vec[i]=ret[i];
    for(int i=0;i<seq_size;i++) ret[i]=0;

    generate_temperory_variables();
}

void free_memmory()
{
    delete col_idx;
    delete row_ptr;
    delete vec_ptr;
    delete val;

    delete spv8_list_lens;
    delete spv8_lists;
    delete spv8_panelsize;

    delete vec;
    delete ret;
    delete ans;

    delete seq;
    delete rseq;
    delete sseq;
}

inline void read_parameters(int argc,char *argv[])
{
    threadnum = argc>=3?atoi(argv[1]):1;
    if(threadnum <= 0)
        threadnum = 1;
    run_num = argc>=4?atoi(argv[3]):1;
}

inline void print_running_time(auto begin, auto end)
{
    auto duration = duration_cast<microseconds>(end - begin);
    double total = double(duration.count());
    double average = total / run_num;
    double average_wb = time_wb / run_num;
    cout << "total cost:" << total<<"in"<<run_num<<"times"<<endl;
    cout << "average cost:" << average << endl;
    cout << "wb's propotion:"<<time_wb/total<<endl;
    cout << "wb's cost:"<<average_wb<<endl;
    cout << "total wb's cost:" << time_wb << endl;
}

inline void export_result(const char * file_name)
{
    // 因为稀疏矩阵经过了bitmap处理和V8处理，行原来的顺序被打乱了，生成的Y向量不是原顺序的，需要进行重排
    for(int i=0;i<row;i++){
        ans[i] = ret[rseq[i]];
    }
    ofstream fout(file_name);
    for (int  i = 0; i < row; ++i)
        fout << (ans[i])<<endl;
    fout.close();
}

int main(int argc,char *argv[])
{
    read_parameters(argc, argv); // 获取外参数，即执行本次运算的运算核数目与迭代次数

    read_data(string(argv[2])); // 获取矩阵参数

    for (int j = 0; j < warm_num; j++) // 先进行300次运算预热，减少cache的cold miss
        spmv_kernel();

    auto begin = high_resolution_clock::now();
    for (int j = 0; j < run_num; j++)
        spmv_kernel();
    auto end = high_resolution_clock::now();

    print_running_time(begin, end);
    
    export_result("result_spmm.txt");

    free_memmory();

    return 0;
}

