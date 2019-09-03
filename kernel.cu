#include "FloydWarshall.hpp"
#include "Graph/GraphWeight.hpp"
#include <tuple>
#include <limits>
#include <iomanip>
#include <iostream>
#include <random>
#include <chrono>

#include <thread>

#include <fstream>


using matrix_t = float; //matrix_t is like float

//TIME
using namespace std;
using namespace std::chrono;

//nV => numero di vettori del grafo
//nE => numero di archi nel grafo


const int BLOCK_SIZE = 16;

const float INF = std::numeric_limits<float>::infinity();


template<typename T> __global__ void floyd_warshall_kernel(T* matrix, int num_vertices, int k) {

    int id = blockIdx.x * blockDim.x + threadIdx.x;  //colonne

    if(id<num_vertices*num_vertices){

        int Y = id/num_vertices;
        int X = id - Y*num_vertices;

        if (matrix[Y*num_vertices + k] != INF &&
            matrix[k*num_vertices + X] != INF &&
            matrix[Y*num_vertices + k] + matrix[k*num_vertices + X] < matrix[Y*num_vertices+X])
                matrix[Y*num_vertices+X] = matrix[Y*num_vertices + k] + matrix[k*num_vertices + X];
    }
}

__host__ int main(int argc, char* argv[]) {


    // se non trova due argomenti termina perchè manca l'input
    if (argc != 2)
        return EXIT_FAILURE;

    // -------------------------------------------------------------------------
    // HOST MEMORY ALLOCATION
    graph::GraphWeight<int, int, matrix_t> graph(graph::structure_prop::COO);   // arco da int A a int B di peso float P
    graph.read(argv[1]);    //si copia da file i valori dati

    auto matrix = new matrix_t*[graph.nV()];    //usata per la parte di codice sequenziale

    // -------------------------------------------------------------------------
    // HOST INITILIZATION
    for (int i = 0; i < graph.nV(); i++) {
        matrix[i] = new matrix_t[graph.nV()];
        std::fill(matrix[i], matrix[i] + graph.nV(), std::numeric_limits<matrix_t>::infinity());    //inizializza tutti gli archi a INF
    }
    for (int i = 0; i < graph.nE(); i++) {
        auto index = graph.coo_ptr()[i];
        matrix[std::get<0>(index)][std::get<1>(index)] = std::get<2>(index);
    }

    //--------------------------------------------------------------------------
    // HOST EXECUTIION

    float* h_matrix = new float[graph.nV()*graph.nV()];

    for (int i = 0; i < graph.nV(); i++)
        for(int j = 0; j < graph.nV(); j++)
            h_matrix[i*graph.nV() + j] = matrix[i][j];

    auto t1 = std::chrono::system_clock::now();
    //std::this_thread::sleep_for(seconds(5));
    floyd_warshall::floyd_warshall(matrix, graph.nV()); //codice sequenziale
    auto t2 = std::chrono::system_clock::now();
    long double duration = duration_cast<milliseconds>( t2 - t1 ).count();
    std::cout << "TIME CPU " << duration/1000 << '\n';

    // -------------------------------------------------------------------------
    // DEVICE MEMORY ALLOCATION
    float *d_matrix;
    int dim_data = graph.nV()*graph.nV()*sizeof(float);
    cudaMalloc(&d_matrix, dim_data);

    // -------------------------------------------------------------------------
    // COPY DATA FROM HOST TO DEVIE
    cudaMemcpy(d_matrix , h_matrix, dim_data, cudaMemcpyHostToDevice);

    // -------------------------------------------------------------------------
    // DEVICE EXECUTION

    //dim3 block_size(BLOCK_SIZE_X,BLOCK_SIZE_Y);
    //dim3 num_blocks( (graph.nV()+BLOCK_SIZE_X-1)/BLOCK_SIZE_X, (graph.nV()+BLOCK_SIZE_Y-1)/BLOCK_SIZE_Y);

    /// time start
    cudaEvent_t startTimeCuda, stopTimeCuda;
    cudaEventCreate(&startTimeCuda);
    cudaEventCreate(&stopTimeCuda);
    cudaEventRecord(startTimeCuda, 0);

    for(int k = 0; k < graph.nV(); k++)
        floyd_warshall_kernel<<< (graph.nV()+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE >>>(d_matrix,graph.nV(), k);

    cudaEventRecord(stopTimeCuda,0);
    cudaEventSynchronize(stopTimeCuda);
    float msTime;
    cudaEventElapsedTime(&msTime, startTimeCuda, stopTimeCuda);

    std::cout << "TIME GPU " << msTime/1000 << '\n';
    std::cout << "SPEEDUP " << (duration/1000)/(msTime/1000) << '\n';

    // -------------------------------------------------------------------------
    // COPY DATA FROM DEVICE TO HOST
    cudaMemcpy(h_matrix, d_matrix, dim_data, cudaMemcpyDeviceToHost);

    // CREATE A FILE OF VALUES
    std::ofstream myfile;
    myfile.open ("../data.csv");
    for (int i = 0; i < graph.nV(); i++){
        for(int j = 0; j < graph.nV(); j++){
            myfile << h_matrix[i*graph.nV() + j] << ",";
        }
        myfile << '\n';
    }
    myfile.close();

    //--------------------------------------------------------------------------
    // RESULT CHECK
    for (int i = 0; i < graph.nV(); i++)
        for(int j = 0; j < graph.nV(); j++)
            if (h_matrix[i*graph.nV()+j] != matrix[i][j]) {
                std::cerr << "wrong result at: ("
                        << i << ", " << j << ")"
                        << "\nhost:   " << matrix[i][j]
                        << "\ndevice: " << h_matrix[i*graph.nV()+j] << "\n\n";
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
