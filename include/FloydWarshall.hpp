#pragma once

namespace floyd_warshall {

template<typename T>
void floyd_warshall(T** matrix, int num_vertices);

template<typename T>
void floyd_warshall_omp(T** matrix, int num_vertices);
} // floyd_warshall
