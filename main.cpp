#include "FloydWarshall.hpp"
#include "Graph/GraphWeight.hpp"
#include <tuple>
#include <limits>
#include <typeinfo>
#include <math.h>

#include <chrono>

using matrix_t = float;

const float INF = std::numeric_limits<matrix_t>::infinity();

int main(int argc, char* argv[]) {
    if (argc != 2)
        return EXIT_FAILURE;

    graph::GraphWeight<int, int, matrix_t> graph(graph::structure_prop::COO);
    graph.read(argv[1]);

    auto matrix = new matrix_t*[graph.nV()];
    for (int i = 0; i < graph.nV(); i++) {
        matrix[i] = new matrix_t[graph.nV()];
        std::fill(matrix[i], matrix[i] + graph.nV(), std::numeric_limits<matrix_t>::infinity());    //inizializza tutti gli archi a INF
                //std::cout << *matrix[i] << " AND " << *matrix[i] + graph.nV() << '\n';

    }
    for (int i = 0; i < graph.nE(); i++) {
        auto index = graph.coo_ptr()[i];
        matrix[std::get<0>(index)][std::get<1>(index)] = std::get<2>(index);
    }


    // create an array data for vertex matrix
    auto h_matrix = new matrix_t*[graph.nV()];
    for (int i = 0; i < graph.nV(); i++){
        h_matrix[i] = new matrix_t[graph.nV()];
        for(int j = 0; j < graph.nV(); j++){
            //if(matrix[i][j] != INF)
            h_matrix[i][j] = matrix[i][j];
        }
    }
    //--------------------------------------------------------------------------

    auto t_start_sq = std::chrono::high_resolution_clock::now();
    floyd_warshall::floyd_warshall(matrix, graph.nV());
    auto t_end_sq = std::chrono::high_resolution_clock::now();

    double elaspedTimeMs_sq = std::chrono::duration<double, std::milli>(t_end_sq-t_start_sq).count();


    auto t_start_pr= std::chrono::high_resolution_clock::now();
    floyd_warshall::floyd_warshall_omp(h_matrix, graph.nV());
    auto t_end_pr = std::chrono::high_resolution_clock::now();
    double elaspedTimeMs_pr = std::chrono::duration<double, std::milli>(t_end_pr-t_start_pr).count();

    std::cout << "TIME OF SEQUENTIAL CODE: " << elaspedTimeMs_sq/1000 << '\n' <<
                 "TIME OF PARALEL CODE (OpenMP): " << elaspedTimeMs_pr/1000 << '\n' <<
                 "SPEEDUP: " << elaspedTimeMs_sq/elaspedTimeMs_pr << "x" << '\n';
    //--------------------------------------------------------------------------
    // RESULT CONTROLL
    bool br;
    for (int i = 0; i < graph.nV(); i++){
        for(int j = 0; j < graph.nV(); j++){
            if(matrix[i][j] != h_matrix[i][j]){
                std::cout << "RISULTATI SBAGLIATI" << '\n';
                br = true;
                break;
            }
        }

        if(br)
            break;
    }

    if(!br)
        std::cout << "CORRETTO" << '\n';

    //--------------------------------------------------------------------------

    for (int i = 0; i < graph.nV(); i++)
        delete[] matrix[i];
    delete[] matrix;

    for (int i = 0; i < graph.nV(); i++)
        delete[] h_matrix[i];
    delete[] h_matrix;
}
