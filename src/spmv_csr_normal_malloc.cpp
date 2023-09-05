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

using namespace std;
using namespace chrono;

int threadnum=1;

int row, col, nnz, seq_size, bnum;

int *col_idx;
int *row_ptr;
int *vec_ptr;

// parameters of spv8
int *spv8_list_lens;
int *spv8_lists;

double *val;
double *vec;
double *new_vec;
double *ret;

int *ans;
int *seq;
int *rseq;
int *sseq;
//是否添加溢出检查？
int *seqreverse_lens;
int *seqreverse;

double time_wb=0;


void write_back(){
	for(int i=0;i<seq_size;i++)
		new_vec[i]=ret[seq[i]];
}

void new_bserial_spmv()//double *ret,double*val,double *vec,int *rowptr,int *seq,int offset=2000)
{
    //vec按serial重排（全是1先不排
    for (int i = 0; i < row; i++) ret[i]=0;

    for(int i=0;i<bnum;i++)//这里写法有待改进（可以考虑用i，每次直接加offset）     
        for(int j=vec_ptr[i];j<vec_ptr[i+1];j++)//对一个块内
            for(int k=row_ptr[j];k<row_ptr[j+1];k++)//对每个行便历rowseq[j]（真正的写回位置，前面是为了便于使用verify进行测试）
                ret[j]+=val[k]*vec[col_idx[k]];
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

void read_data(string route){
    FILE *fp;
    string file_route;
    ifstream inFile;

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
    vec = new double [max(seq_size,row)];
    if (!inFile) {
        cerr << "Unable to open file datafile.txt";
        exit(1);   // call system to stop
    }
    if (inFile) for (int i = 0;i < row;i++) inFile >> vec[i];
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
    rseq = new int [row];
    if (!inFile) {
        cerr << "Unable to open file datafile.txt";
        exit(1);   // call system to stop
    }
    if (inFile) for (int i = 0;i < row;i++) inFile >> rseq[i];
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
        // cerr << "Unable to open file datafile.txt";
        // exit(1);   // call system to stop
    }
    inFile >> count;
    spv8_list_lens = new int [count];
    if (inFile) for (int i = 0;i < count;i++) inFile >> spv8_list_lens[i];
    if (inFile) inFile.close();

    // int count = 0;
    inFile.open(route + "/spv8_list.txt");
    if (!inFile) {
        // cerr << "Unable to open file datafile.txt";
        // exit(1);   // call system to stop
    }
    inFile >> count;
    spv8_lists = new int [count];
    if (inFile) for (int i = 0;i < count;i++) inFile >> spv8_lists[i];
    if (inFile) inFile.close();

    inFile.open(route + "/seq.txt");
    seq = new int [seq_size];
    if (!inFile) {
        cerr << "Unable to open file datafile.txt";
        exit(1);   // call system to stop
    }
    if (inFile) for (int i = 0;i < seq_size;i++) inFile >> seq[i];
    inFile.close();

    new_vec = new double [seq_size];
    ret = new double [max(seq_size,row)];
    ans = new int [row];

    for (int i = 0;i<bnum+1;i++) cout<<vec_ptr[i]<<" ";
}

void free_memmory()
{
    delete col_idx;
    delete row_ptr;
    delete vec_ptr;

    delete spv8_list_lens;
    delete spv8_lists;

    delete val;
    delete vec;
    delete new_vec;
    delete ret;

    delete ans;
    delete seq;
    delete rseq;
    delete sseq;
}

int main(int argc,char *argv[])
{

    read_data(string(argv[2]));

    //如果为true，则进行预热cache及bp等
    int warm_num = argc>=3?atoi(argv[1]):0;

    //修改部分：加入了运行次数的控制
    int run_num = argc>=4?atoi(argv[3]):1;

    // memset(ret,0,sizeof(ret));
    //用ret少开辟一块内存
    cout<<"size is "<<seq_size<<endl;
    for(int i=0;i<seq_size;i++) ret[i]=vec[sseq[i]];
    for(int i=0;i<seq_size;i++) vec[i]=ret[i];

    for (int j = 0; j < warm_num; j++)
        new_bserial_spmv();
    
    auto begin = high_resolution_clock::now();
    for (int j = 0; j < run_num; j++)
        new_bserial_spmv();
    auto end = high_resolution_clock::now();
    
    
    for(int i=0;i<col;i++){
        ans[i]=ret[rseq[i]];
    }

    auto duration = duration_cast<microseconds>(end - begin);
    double total = double(duration.count());
    double average = total / run_num;
    double average_wb = time_wb / run_num;
    cout << "total cost:" << total<<"in"<<run_num<<"times"<<endl;
    cout << "average cost:" << average << endl;
    cout << "wb's propotion:"<<time_wb/total<<endl;
    cout << "wb's cost:"<<average_wb<<endl;
    cout << "total wb's cost:" << time_wb << endl;

    
    ofstream fout("result_spmm.txt");
    for (int  i = 0; i < row; ++i)
    {
      fout << (ans[i])<<endl;
    }
    fout.close();
     
    
    ofstream write;
    write.open("wb_time.txt",ios::app);
    write<<time_wb/total<<endl;
    write.close();
    
    free_memmory();

    return 0;
}

