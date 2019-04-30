#include "FloydWarshall.hpp"
#include "Graph/GraphWeight.hpp"
#include <tuple>
#include <limits>

using matrix_t = float; //matrix_t is like float

//nV => numero di vettori del grafo
//nE => numero di archi nel grafo

const int BLOCK_SIZE_X = 16;
const int BLOCK_SIZE_Y = 16;

__global__
void matrixMultiplicationKernel() {
    unsigned int X = blockIdx.x * blockDim.x +threadIdx.x;
    unsigned int Y = blockIdx.y * blockDim.y +threadIdx.y;
    if(X<N && Y<N){
        unsigned int tmp = 0;
        //printf("%i %i \n", X,Y);
        for(int i=0; i<N; i++){
            tmp+= d_matrixA[i+Y*N]*d_matrixB[X+i*N];
        }
        d_matrixC[X+Y*N]=tmp;
    }

}

int main(int argc, char* argv[]) {

    // se non trova due argomenti termina perchè manca l'input
    if (argc != 2)
        return EXIT_FAILURE;

    // -------------------------------------------------------------------------
    // HOST MEMORY ALLOCATION
    graph::GraphWeight<int, int, matrix_t> graph(graph::structure_prop::COO);   // arco da int A a int B di peso float P 
    graph.read(argv[1]);    //si copia da file i valori dati

    auto matrix = new matrix_t*[graph.nV()];

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
    Timer<DEVICE> TM_device;
    Timer<HOST>   TM_host;
    
    TM_host.start();
    floyd_warshall::floyd_warshall(matrix, graph.nV()); //codice sequenziale
    
    TM_host.stop();
    TM_host.print("FloydWarshall host:   ");

    // create an array data for vertex matrix(solo nel progetto)
    auto h_matrix = new matrix_t[graph.nV()*graph.nV()];
    for (int i = 0; i < graph.nV(); i++)
        for(int j = 0; j < graph.nV(); j++)
            h_matrix[i*graph.nV() + j] = matrix[i][j];

    // -------------------------------------------------------------------------
    // DEVICE MEMORY ALLOCATION
    auto *d_matrix;
    int dim_data = graph.nV()*graph.nV()*sizeof(float);
    cudaMalloc(&d_matrix, dim_data);

    // -------------------------------------------------------------------------
    // COPY DATA FROM HOST TO DEVIE
    cudaMemcpy(d_matrix , h_matrix, dim_data, cudaMemcpyHostToDevice);

    // -------------------------------------------------------------------------
    // DEVICE EXECUTION
    TM_device.start();

    dim3 block_size(BLOCK_SIZE_X,BLOCK_SIZE_Y);

    dim3 num_blocks( (graph.nV()+BLOCK_SIZE_X-1)/BLOCK_SIZE_X, (graph.nV()+BLOCK_SIZE_Y-1)/BLOCK_SIZE_Y );

    matrixMultiplicationKernel<<< num_blocks, block_size >>>(d_matrix,graph.nV());

    TM_device.stop();
    TM_device.print("MatrixMultiplication device: ");

    std::cout << std::setprecision(1)
              << "Speedup: " << TM_host.duration() / TM_device.duration()
              << "x\n\n";

    // -------------------------------------------------------------------------
    // COPY DATA FROM DEVICE TO HOST
    cudaMemcpy(h_matrix , d_matrix, dim_data, cudaMemcpyDeviceToHost);

    //--------------------------------------------------------------------------
    for (int i = 0; i < graph.nV(); i++)
        delete[] matrix[i];
    delete[] matrix;
}
