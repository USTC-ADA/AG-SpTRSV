#ifndef YY_SPTRSV__
#define YY_SPTRSV__

#include "GPU_setup.h"
#include <stdio.h>

#define WARP_PER_BLOCK 32

#define VALUE_TYPE float

__global__ void yySpTRSV_csr_kernel(const int* __restrict__        d_csrRowPtr,
                         const int* __restrict__        d_csrColIdx,
                         const VALUE_TYPE* __restrict__ d_csrVal,
                         int*                           d_get_value,
                         const int                      m,
                         const int                      nnz,
                         const VALUE_TYPE* __restrict__ d_b,
                         VALUE_TYPE*                    d_x,
                         const int                      begin,
                         const int* __restrict__        d_warp_num,
                         const int                      Len,
                         int*                           d_id_extractor)

{
//    const int global_id =atomicAdd(d_id_extractor, 1) - 1;
    const int global_id = (begin + blockIdx.x) * blockDim.x + threadIdx.x;

    const int warp_id = global_id/WARP_SIZE;
    int row;

    if(warp_id>=(Len-1))
        return;

    const int lane_id = (WARP_SIZE - 1) & threadIdx.x;
    
    /* Thread-level Syncfree SpTRSV */
    if(d_warp_num[warp_id+1]>(d_warp_num[warp_id]+1))
    {
        row =d_warp_num[warp_id]+lane_id;
        if(row>=m)
            return;
        
        int col,j,i;
        VALUE_TYPE xi;
        VALUE_TYPE left_sum=0;
        //i是行号，j是当前处理的非零元编号
        i=row;
        j=d_csrRowPtr[i];
        
        while(j<d_csrRowPtr[i+1])
        {
            col=d_csrColIdx[j];
            //while(d_get_value[col]==1)
            __threadfence();
            if(d_get_value[col]==1)
            {
                left_sum+=d_csrVal[j]*d_x[col];
                j++;
                col=d_csrColIdx[j];
            }
            if(i==col)
            {
                xi = (d_b[i] - left_sum) / d_csrVal[d_csrRowPtr[i+1]-1];
                d_x[i] = xi;
                __threadfence();
                d_get_value[i]=1;
                j++;
            }
        }
    }
    else  /* Warp-level Syncfree SpTRSV */
    {
        row = d_warp_num[warp_id];
        if(row>=m)
            return;
        
        int col,j;
        VALUE_TYPE xi,sum=0;
        for (j = d_csrRowPtr[row]  + lane_id; j < d_csrRowPtr[row+1]-1; j += WARP_SIZE)
        {
            
            col=d_csrColIdx[j];
            while(d_get_value[col]!=1)
            {
                __threadfence();
            }
            sum += d_x[col] * d_csrVal[j];
        }
        
        //规约求和
        for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
        
        if (!lane_id)   /* thread 0 in each warp*/
        {
            xi = (d_b[row] - sum) / d_csrVal[d_csrRowPtr[row+1]-1];
            d_x[row]=xi;
            __threadfence();
            d_get_value[row]=1;
        }
        
    }

}

#undef VALUE_TYPE

#define VALUE_TYPE double

__global__ void yySpTRSV_csr_kernel(const int* __restrict__        d_csrRowPtr,
                         const int* __restrict__        d_csrColIdx,
                         const VALUE_TYPE* __restrict__ d_csrVal,
                         int*                           d_get_value,
                         const int                      m,
                         const int                      nnz,
                         const VALUE_TYPE* __restrict__ d_b,
                         VALUE_TYPE*                    d_x,
                         const int                      begin,
                         const int* __restrict__        d_warp_num,
                         const int                      Len,
                         int*                           d_id_extractor)

{
//    const int global_id =atomicAdd(d_id_extractor, 1) - 1;
    const int global_id = (begin + blockIdx.x) * blockDim.x + threadIdx.x;

    const int warp_id = global_id/WARP_SIZE;
    int row;

    if(warp_id>=(Len-1))
        return;

    const int lane_id = (WARP_SIZE - 1) & threadIdx.x;
    
    /* Thread-level Syncfree SpTRSV */
    if(d_warp_num[warp_id+1]>(d_warp_num[warp_id]+1))
    {
        row =d_warp_num[warp_id]+lane_id;
        if(row>=m)
            return;
        
        int col,j,i;
        VALUE_TYPE xi;
        VALUE_TYPE left_sum=0;
        //i是行号，j是当前处理的非零元编号
        i=row;
        j=d_csrRowPtr[i];
        
        while(j<d_csrRowPtr[i+1])
        {
            col=d_csrColIdx[j];
            //while(d_get_value[col]==1)
            __threadfence();
            if(d_get_value[col]==1)
            {
                left_sum+=d_csrVal[j]*d_x[col];
                j++;
                col=d_csrColIdx[j];
            }
            if(i==col)
            {
                xi = (d_b[i] - left_sum) / d_csrVal[d_csrRowPtr[i+1]-1];
                d_x[i] = xi;
                __threadfence();
                d_get_value[i]=1;
                j++;
            }
        }
    }
    else  /* Warp-level Syncfree SpTRSV */
    {
        row = d_warp_num[warp_id];
        if(row>=m)
            return;
        
        int col,j;
        VALUE_TYPE xi,sum=0;
        for (j = d_csrRowPtr[row]  + lane_id; j < d_csrRowPtr[row+1]-1; j += WARP_SIZE)
        {
            
            col=d_csrColIdx[j];
            while(d_get_value[col]!=1)
            {
                __threadfence();
            }
            sum += d_x[col] * d_csrVal[j];
        }
        
        //规约求和
        for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
        
        if (!lane_id)   /* thread 0 in each warp*/
        {
            xi = (d_b[row] - sum) / d_csrVal[d_csrRowPtr[row+1]-1];
            d_x[row]=xi;
            __threadfence();
            d_get_value[row]=1;
        }
        
    }

}

#undef VALUE_TYPE

template <typename T>
int matrix_layer(const int         m,
                 const int         n,
                 const int         nnz,
                 const int        *csrRowPtr,
                 const int        *csrColIdx,
                 int              *layer_add,
                 double           *parallelism_add
                 )

{
    int *layer=(int *)malloc(m*sizeof(int));
    if (layer==NULL)
        printf("layer error\n");
    memset (layer, 0, sizeof(int)*m);

    int *layer_num=(int *)malloc((m+1)*sizeof(int));
    if (layer_num==NULL)
        printf("layer_num error\n");
    memset (layer_num, 0, sizeof(int)*(m+1));
    
    int max_layer;
    int max_layer2=0;
    int max=0;
    unsigned int min=-1;

    // count layer
    int row,j;
    for (row = 0; row < m; row++)
    {
        max_layer=0;
        for (j = csrRowPtr[row]; j < csrRowPtr[row+1]; j++)
        {
            int col = csrColIdx[j];

            if((layer[col]+1)>max_layer)
                max_layer=layer[col]+1;

        }
        layer[row]=max_layer;
        layer_num[max_layer]++;
        if (max_layer>max_layer2)
            max_layer2=max_layer;
    }
    for(j=1;j<=max_layer2;j++)
    {
        if(max<layer_num[j])
            max=layer_num[j];
        if(min>layer_num[j])
            min=layer_num[j];
    }

    double avg=(double)m/max_layer2;
    free(layer);
    free(layer_num);

    //printf("matrix L's layer = %d, average numer of nodes in layer = %d\n",max_layer2,avg);
    //int min2=min;
    //printf("the minimun parallelism is %d,the maximun parallelism is %d\n",min2,max);
    *layer_add=max_layer2;
    *parallelism_add=avg;
    //printf(",%d,%d,%d",nnz,max_layer2,avg);
    return max_layer2;

}

template <typename T>
int matrix_layer2(const int         m,
                 const int         n,
                 const int         nnz,
                 const int        *csrRowPtr,
                 const int        *csrColIdx,
                 int              *layer_add,
                 double           *parallelism_add,
                 int              *max_row_nnz
                 )

{
    int *layer=(int *)malloc(m*sizeof(int));
    if (layer==NULL)
        printf("layer error\n");
    memset (layer, 0, sizeof(int)*m);

    int *layer_num=(int *)malloc((m+1)*sizeof(int));
    if (layer_num==NULL)
        printf("layer_num error\n");
    memset (layer_num, 0, sizeof(int)*(m+1));
    
    int max_layer;
    int max_layer2=0;
    int max=0;
    unsigned int min=-1;

    *max_row_nnz = 0;

    // count layer
    int row,j;
    for (row = 0; row < m; row++)
    {
        max_layer=0;
        for (j = csrRowPtr[row]; j < csrRowPtr[row+1]; j++)
        {
            int col = csrColIdx[j];

            if((layer[col]+1)>max_layer)
                max_layer=layer[col]+1;

        }

        if (*max_row_nnz < csrRowPtr[row+1] - csrRowPtr[row])
            *max_row_nnz = csrRowPtr[row+1] - csrRowPtr[row];

        layer[row]=max_layer;
        layer_num[max_layer]++;
        if (max_layer>max_layer2)
            max_layer2=max_layer;
    }
    //printf("layer_num: ");
    for(j=1;j<=max_layer2;j++)
    {
        if(max<layer_num[j])
            max=layer_num[j];
        if(min>layer_num[j])
            min=layer_num[j];
        //printf("%d ", layer_num[j]);
    }
    //printf("\n");

    double avg=(double)m/max_layer2;
    free(layer);
    free(layer_num);

    //printf("matrix L's layer = %d, average numer of nodes in layer = %d\n",max_layer2,avg);
    //int min2=min;
    //printf("the minimun parallelism is %d,the maximun parallelism is %d\n",min2,max);
    *layer_add=max_layer2;
    *parallelism_add=avg;
    //printf(",%d,%d,%d",nnz,max_layer2,avg);
    return max_layer2;

}

template <typename T>
void matrix_warp     (const int         m,
                      const int         n,
                      const int         nnz,
                      const int        *csrRowPtr,
                      const int        *csrColIdx,
                      const T          *csrVal,
                      const int         border,
                      int              *Len_add,
                      int              *warp_num,
                      double           *warp_occupy_add,
                      double           *element_occupy_add
                      )
{
    int end;
    int i;
    int element_n=0;
    double avg_element_n=0;
    int warp_greater=0,warp_lower=0;
    double element_warp=0,row_warp=0;
    int row=0;
    warp_num[0]=0;
    int k=1,j;
    for(i=0;i<m;i=i+WARP_SIZE)
    {
        end = i+WARP_SIZE;
        if(m<end)
            end=m;
        element_n=csrRowPtr[end]-csrRowPtr[i];
        avg_element_n = ((double)element_n)/(end-i);
        if(avg_element_n>=border)
        {
            warp_greater++;
            element_warp+=(double)element_n;
            row_warp+=(double)(end-i);
            for(j=0;j<(end-i);j++)
            {
                row++;
                warp_num[k]=row;
                k++;
            }
        }
        else
        {
            warp_lower++;
            row += (end-i);
            warp_num[k]=row;
            k++;
        }
    }
    
    int Len=k;
    *Len_add=Len;
    *warp_occupy_add = row_warp/m;
    *element_occupy_add = element_warp/nnz;
}


#endif

