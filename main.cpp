#include "FloydWarshall.hpp"
#include "Graph/GraphWeight.hpp"
#include <tuple>
#include <limits>

using matrix_t = float;

int main(int argc, char* argv[]) {
    if (argc != 2)
        return EXIT_FAILURE;

    graph::GraphWeight<int, int, matrix_t> graph(graph::structure_prop::COO);
    graph.read(argv[1]);

    auto matrix = new matrix_t*[graph.nV()];
    for (int i = 0; i < graph.nV(); i++) {
        matrix[i] = new matrix_t[graph.nV()];
        std::fill(matrix[i], matrix[i] + graph.nV(),
                  std::numeric_limits<matrix_t>::infinity());
    }
    for (int i = 0; i < graph.nE(); i++) {
        auto index = graph.coo_ptr()[i];
        matrix[std::get<0>(index)][std::get<1>(index)] = std::get<2>(index);
    }
    //--------------------------------------------------------------------------

    floyd_warshall::floyd_warshall(matrix, graph.nV());

    //--------------------------------------------------------------------------
    for (int i = 0; i < graph.nV(); i++)
        delete[] matrix[i];
    delete[] matrix;
}
