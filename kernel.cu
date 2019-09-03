#include "FloydWarshall.hpp"
#include "Graph/GraphWeight.hpp"
#include <tuple>
#include <limits>
#include <iomanip>
#include <iostream>
#include <random>
#include <chrono>
#include <algorithm>    // std::max


using matrix_t = float; //matrix_t is like float

//TIME
using namespace std;
using namespace std::chrono;

//nV => numero di vettori del grafo
//nE => numero di archi nel grafo

const int dBS = 32;

__constant__ float dINF;
__constant__ int dnV;
//__constant__ int dBS;

bool cmpFloats2(float a, float b) {
    const float abs_a = std::abs(a);
    const float abs_b = std::abs(b);
    const float epsilon = 0.00001f;  // prende solo 4 valori dopo la virgola
    float diff = std::abs(abs_a - abs_b);
    if(diff / std::max(abs_a, abs_b) > epsilon)
        return true;
    return false;
}

// bool ph indirizza la funzione di calcolo floyd warshall se siamo nella fase 3 o no [ph == true allora siamo nella fase 3]
__forceinline__ __device__ void floyd_warshall_kernel(float* matrixC, float* matrixA, float* matrixB, int Y, int X, bool ph3){

    float dAB;

    if(!ph3){
        for(unsigned int k = 0; k < dBS; k++){
            dAB = matrixA[Y*dBS + k] + matrixB[k*dBS + X];
            if(dAB != dINF && matrixC[Y*dBS + X] > dAB){
                matrixC[Y*dBS + X] = dAB;}
            __syncthreads();
        }
    }

    else{
        for(unsigned int k = 0; k < dBS; k++){
            dAB = matrixA[Y*dBS + k] + matrixB[k*dBS + X];
            if(dAB != dINF &&  matrixC[Y*dBS + X] > dAB)
                matrixC[Y*dBS + X] = dAB;
        }
    }
}


// FASE 1 => BLOCCO PRIMARIO
__global__ void block_ph1(float* matrix, int pbid){

    const unsigned int by = blockIdx.y;
    const unsigned int bx = blockIdx.x;
    
    const unsigned int pb_point = by*dBS;

    const unsigned int Y = threadIdx.y;
    const unsigned int X = threadIdx.x;

    if(by != pbid || bx != pbid)    // se si dovesse trovare sulla diagonale
        return;

    __shared__ float C[dBS*dBS];

    __syncthreads();

    if(pb_point + Y < dnV && pb_point + X < dnV)// non siamo fuori range della matrice
        C[Y*dBS + X] = matrix[pb_point*dnV+pb_point+Y*dnV+X];
    else
        C[Y*dBS + X] = dINF;

    __syncthreads();
    floyd_warshall_kernel(C, C, C, Y, X, false);
    __syncthreads();

    if(pb_point + Y < dnV && pb_point + X < dnV)
        matrix[pb_point*dnV+pb_point+Y*dnV+X] = C[Y*dBS + X];

} 

// FASE 2 => BLOCCHI CON RIGHE O COLONNE IN COMUNE AL BLOCCO PRIMARIO
__global__ void block_ph2(float* matrix, int pbid){

    const unsigned int by = blockIdx.y;
    const unsigned int bx = blockIdx.x;

    const unsigned int by_point = by*dBS;
    const unsigned int bx_point = bx*dBS;
    const unsigned int pb_point = pbid*dBS;

    if((bx == 0 && by == pbid) || (bx == pbid && by == 0))
        return;

    const unsigned int Y = threadIdx.y;
    const unsigned int X = threadIdx.x;

    __shared__ float C[dBS*dBS];
    __shared__ float A[dBS*dBS];
    __shared__ float B[dBS*dBS];

    __syncthreads();
    
    ///////////////////// CB phase 1
    if(by == 0){
        if(pb_point+Y < dnV && bx_point+X < dnV)// non siamo fuori range della matrice
            C[Y*dBS + X] = matrix[pb_point*dnV+bx_point+Y*dnV+X];
        else
            C[Y*dBS + X] = dINF;

        if(pb_point+Y < dnV && pb_point+X < dnV)// non siamo fuori range della matrice
            A[Y*dBS + X] = matrix[pb_point*dnV+pb_point+Y*dnV+X];
        else
            A[Y*dBS + X] = dINF;

        __syncthreads();
        floyd_warshall_kernel(C, A, C, Y, X, false);
        __syncthreads();

        if(pb_point+Y < dnV && bx_point+X < dnV)// non siamo fuori range della matrice
            matrix[pb_point*dnV+bx_point+Y*dnV+X] = C[Y*dBS + X];
    }

        ///////////////////// CA phase 2
    if(bx == 0){

        if(by_point+Y < dnV && pb_point+X < dnV)// non siamo fuori range della matrice
            C[Y*dBS + X] = matrix[by_point*dnV+pb_point+Y*dnV+X];
        else
            C[Y*dBS + X] = dINF;

        if(pb_point+Y < dnV && pb_point+X < dnV)// non siamo fuori range della matrice
            B[Y*dBS + X] = matrix[pb_point*dnV+pb_point+Y*dnV+X];
        else
            B[Y*dBS + X] = dINF;

        __syncthreads();
        floyd_warshall_kernel(C, C, B, Y, X, false);
        __syncthreads();

        if(by_point+Y < dnV && pb_point+X < dnV) //non siamo fuori range della matrice
            matrix[by_point*dnV+pb_point+Y*dnV+X] = C[Y*dBS + X];
    }
}

//  FASE 3 => BLOCCHI RIMANENTI
__global__ void block_ph3(float* matrix, int pbid){

    const unsigned int by = blockIdx.y;
    const unsigned int bx = blockIdx.x;

    const unsigned int by_point = by*dBS;
    const unsigned int bx_point = bx*dBS;
    const unsigned int pb_point = pbid *dBS;

    const unsigned int Y = threadIdx.y;
    const unsigned int X = threadIdx.x;
    
    if(bx == pbid || by == pbid)
        return;

    __shared__ float C[dBS*dBS];
    __shared__ float A[dBS*dBS];
    __shared__ float B[dBS*dBS];

    __syncthreads();

    if(bx_point+X < dnV && by_point+Y < dnV)// non siamo fuori range della matrice
        C[Y*dBS + X] = matrix[by_point*dnV+bx_point+Y*dnV+X];
    else
        C[Y*dBS + X] = dINF;

    if(by_point+Y < dnV &&  pb_point+X < dnV)// non siamo fuori range della matrice
        A[Y*dBS + X] = matrix[by_point*dnV+pb_point+Y*dnV+X];
    else
        A[Y*dBS + X] = dINF;

    if(pb_point+Y < dnV && bx_point+X < dnV)// non siamo fuori range della matrice
        B[Y*dBS + X] = matrix[pb_point*dnV+bx_point+Y*dnV+X];
    else
        B[Y*dBS + X] = dINF;

    __syncthreads();
    floyd_warshall_kernel(C, A, B, Y, X, true);
    __syncthreads();

    if(by_point+Y < dnV && bx_point+X < dnV)// non siamo fuori range della matrice
        matrix[by_point*dnV+bx_point+Y*dnV+X] = C[Y*dBS + X];
} 

// CODICE LANCIATO SULLA CPU
__host__ int main(int argc, char* argv[]) {

    int count = 0;
        cudaGetDeviceCount(&count);

    // se non trova due argomenti termina perchè manca l'input
    if (argc != 2)
        return EXIT_FAILURE;

    // -------------------------------------------------------------------------
    // HOST MEMORY ALLOCATION
    graph::GraphWeight<int, int, matrix_t> graph(graph::structure_prop::COO);   // arco da int A a int B di peso float P 
    graph.read(argv[1]);    //si copia da file i valori dati

    const unsigned int nV = graph.nV();
    const float INF = std::numeric_limits<float>::infinity();

    auto matrix = new matrix_t*[nV];    //usata per la parte di codice sequenziale

    // -------------------------------------------------------------------------
    // HOST INITILIZATION
    for (int i = 0; i < nV; i++) {
        matrix[i] = new matrix_t[nV];
        std::fill(matrix[i], matrix[i] + nV, std::numeric_limits<matrix_t>::infinity());    //inizializza tutti gli archi a INF
    }
    for (int i = 0; i < graph.nE(); i++) {
        auto index = graph.coo_ptr()[i];
        matrix[std::get<0>(index)][std::get<1>(index)] = std::get<2>(index);
    }

    //--------------------------------------------------------------------------
    // HOST EXECUTIION

    float* h_matrix = new float[nV*nV];

    for (int i = 0; i < nV; i++)
        for(int j = 0; j < nV; j++)
            if(i < nV && j < nV)
                h_matrix[i*nV + j] = matrix[i][j];

    auto t1 = std::chrono::system_clock::now();
    floyd_warshall::floyd_warshall(matrix, nV); //codice sequenziale
    auto t2 = std::chrono::system_clock::now();
    long double duration = duration_cast<milliseconds>( t2 - t1 ).count();
    std::cout << "TIME CPU " << duration/1000 << '\n';

    // -------------------------------------------------------------------------
    // DEVICE MEMORY ALLOCATION
    float *d_matrix;
    const unsigned int dim_data = nV*nV*sizeof(float);
    cudaMalloc(&d_matrix, dim_data);


    // -------------------------------------------------------------------------
    // COPY DATA FROM HOST TO DEVIE
    cudaMemcpy(d_matrix, h_matrix, dim_data, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(dINF, &INF, sizeof(float), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(dnV, &nV, sizeof(int), 0, cudaMemcpyHostToDevice);
    //cudaMemcpyToSymbol(dBS, &BLOCK_SIZE, sizeof(int), 0, cudaMemcpyHostToDevice);


    // -------------------------------------------------------------------------
    // DEVICE EXECUTION
    //const unsigned int dim_sblock = ceil((float) nV / (float) BLOCK_SIZE);
    const unsigned int nblock =  (nV+dBS-1)/dBS;
    //dim3 num_blocksph1(1,1);
    dim3 block_size(dBS,dBS);
    dim3 num_blocks(nblock, nblock);


    /// time start
    cudaEvent_t startTimeCuda, stopTimeCuda;
    cudaEventCreate(&startTimeCuda);
    cudaEventCreate(&stopTimeCuda);
    cudaEventRecord(startTimeCuda);

    //cudaError_t error;

    for(int primary_block = 0; primary_block < nblock; primary_block++){
        //std::cout << "Iterazione numero " << primary_block << '\n';
        block_ph1<<<num_blocks, block_size>>>(d_matrix, primary_block);
        //error = cudaGetLastError();
        //std::cout << "PHASE 1: " << error << '\n';
        block_ph2<<<num_blocks, block_size>>>(d_matrix, primary_block);
        //error = cudaGetLastError();
        //std::cout << "PHASE 2: "  << error << '\n';
        block_ph3<<<num_blocks, block_size>>>(d_matrix, primary_block);
        //error = cudaGetLastError();
        //std::cout << "PHASE 3: "  << error << '\n';
    }

    // TAKE KERNEL EXECUTION TIME
    //cudaDeviceSynchronize();
    cudaEventRecord(stopTimeCuda);
    cudaEventSynchronize(stopTimeCuda);
    float msTime;
    cudaEventElapsedTime(&msTime, startTimeCuda, stopTimeCuda);
    cudaEventDestroy(startTimeCuda);
    cudaEventDestroy(stopTimeCuda);

    std::cout << "TIME GPU " << msTime/1000 << '\n';
    std::cout << "SPEEDUP " << (duration/1000)/(msTime/1000) << '\n';

    // -------------------------------------------------------------------------
    // COPY DATA FROM DEVICE TO HOST
    cudaMemcpy(h_matrix, d_matrix, dim_data, cudaMemcpyDeviceToHost);


    //--------------------------------------------------------------------------
    // RESULT CHECK

    for (int i = 0; i < nV; i++)
        for(int j = 0; j < nV; j++)
            if( cmpFloats2(h_matrix[i*nV+j], matrix[i][j])) {
                std::cerr << "wrong result at: ("
                    << i << ", " << j << ")"
                    << "\nhost:   " << std::setprecision (15) << matrix[i][j]
                    << "\ndevice: " << std::setprecision (15) << h_matrix[i*nV+j] << "\n\n";  
                cudaDeviceReset();
                std::exit(EXIT_FAILURE);
        }

    std::cout << "<> Correct\n\n";

    // -------------------------------------------------------------------------
    // HOST MEMORY DEALLOCATION
    delete[] h_matrix;
    delete[] matrix;
    

    // -------------------------------------------------------------------------
    // DEVICE MEMORY DEALLOCATION
    cudaFree(d_matrix);

    // -------------------------------------------------------------------------
    cudaDeviceReset();
}