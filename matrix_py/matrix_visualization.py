from PIL import Image
import scipy
import scipy.io
import scipy.sparse
import numpy as np
import os


def rdmtx(a):
    matrix = scipy.io.mmread(a)
    mr = scipy.sparse.csr_matrix(matrix)
    print("mtx read: ", a)
    return mr

def gen_img(mr, width, height, normalized=True, all_or_none=False, fix_scale=False,dark=True):#可以在这里选择模式，或者调用时设置
    img = Image.new('L', (width, height), (0))
    mat = np.array(img)
    matrix=np.zeros((height,width),dtype=int)
    rstep = (mr.shape[0]//(height))+1
    cstep = (mr.shape[1]//(width))+1
    print('Pixel Size={}x{}'.format(rstep,cstep))
    # print(cstep)

    for i in range(mr.shape[0]):
        #print(i)
        for j in range(mr.indptr[i], mr.indptr[i+1]):
            matrix[i//rstep][mr.indices[j]//cstep] = matrix[i//rstep][mr.indices[j]//cstep] + 1
    m=matrix.max()
    print('Highest Pixel={}'.format(m))
    if m > rstep*cstep//20:
        m = rstep*cstep//20

    if normalized:
        matrix=matrix * 255 // m
        print("Render with: normalized")
    if all_or_none:
        matrix=(matrix >= 1000) * 255
        print("Render with: all_or_none")
    if fix_scale:
        matrix=(matrix * 128 * 256) // (m)
        print("Render with: fix_scale")

    for i in range (width):
        for j in range (height):
            if matrix[i][j]>255:
                matrix[i][j]=255
    if dark:
        for i in range (width):
            for j in range (height):
                if matrix[i][j]>0 and matrix[i][j]<24:
                    matrix[i][j]=0
        print("Light up dark pixels")

    for i in range(width):
        for j in range(height):
            mat[i][j] =matrix[i][j]
    #mat = matrix.astype(int)#不知道为什么格式转换不能用传播

    img = Image.fromarray(mat)
    print(mat)
    # img.show()
    # img.save('/home/xiatian/Work/spmm/visual/web-Stanford.png')#这里输入保存的文件的路径，推荐保存成png，保存jpeg似乎需要修改保存的代码
    return img

def main():
    mlist = [   'amazon0312',
                'caidaRouterLevel',
                'citationCiteseer',
                'coAuthorsCiteseer',
                'coAuthorsDBLP',
                'dblp-2010',
                'email-EuAll',
                'internet',
                'kron_g500-logn16',
                'language',
                'patents_main',
                'soc-sign-epinions',
                'sx-askubuntu',
                'sx-superuser',
                'web-Stanford',
                'web-Google'
                ]
    for m in mlist:
        trace='/home/xiatian/Work/spmm/mat/mtx/{}/{}.mtx'.format(m,m)#这里输入需要可视化的mtx的路径
        width=1000#图像长度
        height=1000#图像宽度
        mr=rdmtx(trace)
        img=gen_img(mr,width,height)
        img.save('/home/xiatian/Work/spmm/visual/{}.png'.format(m))#这里输入保存的文件的路径，推荐保存成png，保存jpeg似乎需要修改保存的代码


if __name__ == '__main__':
    main()




