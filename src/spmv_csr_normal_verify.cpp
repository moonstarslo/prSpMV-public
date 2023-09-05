#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <iostream>
#include <fstream>
#include <chrono>
#include "omp.h"
#include <immintrin.h>

#define BLOCK_NUM 8

using namespace std;
using namespace chrono;

// const int max_row = 1000000;
// const int max_col = 1000000;
// const int max_nnz = 50000000;

// const int max_row = 200000;
// const int max_col = 200000;
// const int max_nnz = 1000000;

const int threadnum=1;

int block_num = 0;

int row, col, nnz;
int nnz_vec[BLOCK_NUM];
int row_vec[BLOCK_NUM];
int col_vec[BLOCK_NUM];

// int col_idx[max_nnz];
// int row_ptr[max_row];
int *col_idx;
int *row_ptr;
int *row_ptr_vec[BLOCK_NUM];
int *col_idx_vec[BLOCK_NUM];

int *vec_ptr;

// double val[max_nnz];
// double vec[max_col];
// double temp_vec[max_col];
// double ret[max_row];
// double verify[max_row];
double *val;
double *val_vec[BLOCK_NUM];


double *vec;
double *ret;

// user guide
// 可执行文件 true/false 矩阵路径名称
// true/false 表示是否需要预热cache，分支预测器
// cmd example: ./spmv_csr_normal.elf true mat/caidaRouterLevel/

void  mul()//计算一次spmv
{        
    for (int i = 0; i < row; i++) ret[i]=0;
    //按行分别计算结果向量中的不同分量

    for (int i = 0; i < row; i++) {
      for (int j = row_ptr[i]; j < row_ptr[i+1]; j++) {
        ret[i] += val[j] * vec[col_idx[j]];
            }
        }

    for(int i=0;i<row;i++) vec[i]=ret[i];
}

void mul_split_blocks()
{
    omp_set_num_threads(threadnum);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < row; i++) ret[i]=0;
    //按行分别计算结果向量中的不同分量
    
    // omp_set_num_threads(threadnum);
    // #pragma omp parallel for schedule(dynamic)
    // for(int k = 0; k < block_num; k++)
    //     for (int i = vec_ptr[k]; i < vec_ptr[k+1]; i++) {
    //         for (int j = row_ptr[i]; j < row_ptr[i+1]; j++) {
    //             ret[i] += val[j] * vec[col_idx[j]];
    //                 }
    //             }

    omp_set_num_threads(threadnum);
    #pragma omp parallel for schedule(dynamic)
    for (int k = 0; k < row; k += 2048)
        for (int i = k; i < (row < (k + 2048) ? row : (k + 2048)); i++) {
            for (int j = row_ptr[i]; j < row_ptr[i+1]; j++) {
                ret[i] += val[j] * vec[col_idx[j]];
                    }
                }

    // omp_set_num_threads(threadnum);
    // #pragma omp parallel for schedule(dynamic)
    // for (int i = 0; i < row; i++) {
    //   for (int j = row_ptr[i]; j < row_ptr[i+1]; j++) {
    //     ret[i] += val[j] * vec[col_idx[j]];
    //         }
    //     }

    omp_set_num_threads(threadnum);
    #pragma omp parallel for schedule(static)
    for(int i=0;i<row;i++) vec[i]=ret[i];
}

void split_blocks()
{
    block_num = row / 2048;
    // block_num = threadnum;
    int boundary = nnz / block_num + 1;
    int sum = 0;
    int index = 1;
    vec_ptr = new int [block_num + 1];
    vec_ptr[0] = 0;
    for(int i = 0; i < row; i+=8)
    {
        if(row_ptr[i] > sum + boundary)
        {
            vec_ptr[index] = i;
            sum += boundary;
            index++;
        }
    }
    vec_ptr[block_num] = row;
}

void  accelerate_mul(int first = 0)//计算一次spmv
{        
    if(first)
        for (int i = 0; i < row; i++) vec[i]=0;
    //按行分别计算结果向量中的不同分量
    for (int i = 0; i < row; i++) {
      for (int j = row_ptr[i]; j < row_ptr[i+1]; j++) {
        vec[i] += val[j] * vec[col_idx[j]];
            }
        }

    // if(first)
    //     for(int i=0;i<row;i++) vec[i]=ret[i];
    // else
    //     for(int i=0;i<row;i++) vec[i]+=ret[i];
}

void assign_next_parameters(int index)
{
    row = row_vec[index];
    col = col_vec[index];
    nnz = nnz_vec[index];

    row_ptr = row_ptr_vec[index];
    col_idx = col_idx_vec[index];

    val = val_vec[index];
}

void  mul_warm()//计算一次spmv不写回的spmv
{        
    
    //按行分别计算结果向量中的不同分量
    for (int i = 0; i < row; i++) {
      for (int j = row_ptr[i]; j < row_ptr[i+1]; j++) {
        ret[i] += val[j] * vec[col_idx[j]];
            }
        }
    for (int i = 0; i < row; i++) ret[i]=0;
}

void handle_error(){
    cout<<"Error in loading data."<<endl;
    exit(0);
}

void read_data_blocks(string route)
{
    FILE *fp;
    string file_route;
    ifstream inFile;

    for(int index = 0; index < BLOCK_NUM; index++)
    {
        file_route=route+"/info_" + to_string(index) + ".txt";
        fp=freopen(file_route.c_str(),"r",stdin);
        if (fp) cin>>nnz_vec[index]>>row_vec[index]>>col_vec[index];
        else handle_error();
        fclose(fp);

        inFile.open(route+"/row_" + to_string(index) + ".txt");
        row_ptr_vec[index] = new int [row_vec[index] + 1];
        if (!inFile) {
            cerr << "Unable to open file datafile.txt";
            exit(1);   // call system to stop
        }
        if (inFile) for (int i = 0;i <= row_vec[index];i++) inFile >> row_ptr_vec[index][i];
        inFile.close();

        inFile.open(route+"/col_" + to_string(index) + ".txt");
        col_idx_vec[index] = new int [nnz_vec[index]];
        if (!inFile) {
            cerr << "Unable to open file datafile.txt";
            exit(1);   // call system to stop
        }
        if (inFile) for (int i = 0;i < nnz_vec[index];i++) inFile >> col_idx_vec[index][i];
        inFile.close();

        inFile.open(route+"/nnz_" + to_string(index) + ".txt");
        val_vec[index] = new double [nnz_vec[index]];
        if (!inFile) {
            cerr << "Unable to open file datafile.txt";
            exit(1);   // call system to stop
        }
        if (inFile) for (int i = 0;i < nnz_vec[index];i++) inFile >> val_vec[index][i];
        inFile.close();

    }
    // read data from info.txt
    ret = new double [row_vec[0]];


    inFile.open(route + "/x.txt");
    vec = new double [row_vec[0]];
    if (!inFile) {
        cerr << "Unable to open file datafile.txt";
        exit(1);   // call system to stop
    }
    if (inFile) for (int i = 0;i < row_vec[0];i++) inFile >> vec[i];
    inFile.close();
}

void read_data(string route){
    FILE *fp;
    string file_route;

    // read data from info.txt
    file_route=route+"/info.txt";
    fp=freopen(file_route.c_str(),"r",stdin);
    if (fp) cin>>nnz>>row>>col;
    else handle_error();
    fclose(fp);

    // read data from row.txt
    file_route=route+"/row.txt";
    fp=freopen(file_route.c_str(),"r",stdin);
    // row_ptr 的元素个数实际上比 row 的值大1
    row_ptr = new int [row + 1];
    if (fp) for (int i=0;i<=row;i++) cin>>row_ptr[i];
    else handle_error();
    fclose(fp);

    // read data from row.txt
    file_route=route+"/col.txt";
    col_idx = new int [nnz];
    fp=freopen(file_route.c_str(),"r",stdin);
    if (fp) for (int i=0;i<nnz;i++) cin>>col_idx[i];
    else handle_error();
    fclose(fp);

    // read data from nnz.txt
    file_route=route+"/nnz.txt";
    val = new double [nnz];
    fp=freopen(file_route.c_str(),"r",stdin);
    if (fp) for (int i=0;i<nnz;i++) cin>>val[i];
    else handle_error();
    fclose(fp);

    // read data from x.txt
    file_route=route+"/x.txt";
    vec = new double [col];
    fp=freopen(file_route.c_str(),"r",stdin);
    if (fp) for (int i=0;i<col;i++) cin>>vec[i];
    else handle_error();
    fclose(fp);

    ret = new double [row];
    
}

void free_memory()
{
    delete col_idx;
    delete row_ptr;
    delete val;
    delete vec;
    delete ret;
}

int main(int argc,char *argv[])
{
    int blocks_read = argc>=5?atoi(argv[4]):0;
    // 不对矩阵进行列分块
    blocks_read = 0;

    if(blocks_read)
        read_data_blocks(string(argv[2]));
    else
        read_data(string(argv[2]));
    
    //预热cache
    int warm_num = atoi(argv[1]);
    // for(int i=0;i<col;i++) temp_vec[i]=vec[i];
    
    // if (warm_num) printf("the processor is warmed\n");
    // else if (warm_num) {printf("illegal time of warm\n");exit(0);}
    
    // for (int i=0;i<warm_num;i++){
    //         mul();
    //         }
    
    // for (int i = 0; i < row; i++) ret[i]=0;
    // for(int i=0;i<col;i++) vec[i]=temp_vec[i];
    
    // memset(ret,0,sizeof(ret));
    int run_time=atoi(argv[3]);
    if(run_time<=0) {cout<<"illegal time of run"<<endl;exit(0);}
    split_blocks();
    auto begin = high_resolution_clock::now();
    if(blocks_read)
    {
        for(int j=0; j<run_time; j++)
            for(int i = 0; i < BLOCK_NUM; i++)
            {
                assign_next_parameters(i);
                accelerate_mul(i==0);
            }
    }
    else if(threadnum == 1)
        for(int j=0; j<run_time; j++) mul();
    else
        for(int j=0; j<run_time; j++) mul_split_blocks();
    auto end = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(end - begin);
    cout <<"total time: "<< double(duration.count()) << endl;
    cout <<"average time: "<< double(duration.count())/run_time << endl;
    //读入结果并验证正确性，并输出错误部分
    /*read_ans();
    int flag = 1;
    for (int i = 0; i < row; i++)
    {
         if (verify[i] != ret[i])
        {
              flag = 0;
              cout<<verify[i]<<"&&"<<ret[i]<<endl;
          }
      }
    
      if (flag)
      {
          cout<<"Verified"<<endl;
      }
      else
      {
          cout<<"Error in calculation"<<endl;
      }*/
    ofstream fout("result.txt");
    for (int  i = 0; i < row; ++i)
	    fout << ret[i]<<endl;
    fout.close();

    /*cout<<"Answer for SpMV is :"<<endl;
     for (int  i = 0; i < row; ++i)
        cout<<ret[i]<<endl;*/
    free_memory();

    return 0;
}


