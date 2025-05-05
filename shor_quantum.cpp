#include "shor_quantum.h"
#include <omp.h>
#include <cmath>
#include <random>
#include <bitset>
#include <iostream>
#include <chrono>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Gera a matriz Hadamard de n qubits.
// A matriz tem dimensão 2^n x 2^n, usada para criar superposição uniforme de estados.
// Elementos valem 1/sqrt(2^n) multiplicados por +1 ou -1 dependendo dos bits de i & j.
std::vector<std::vector<complexd>> hadamard_all(int n) {
    int dim = 1 << n;
    std::vector<std::vector<complexd>> H(dim, std::vector<complexd>(dim, 1));
    double norm = 1.0 / std::sqrt(dim);
    for (int i = 0; i < dim; ++i)
        for (int j = 0; j < dim; ++j) {
            int bit_count = __builtin_popcount(i & j);
            H[i][j] = norm * ((bit_count % 2 == 0) ? 1 : -1);
        }
    return H;
}

// Gera a matriz da Transformada de Fourier Quântica inversa (QFT^†) para n qubits.
// Cada elemento é dado por exp(-2πi * i * j / N) / sqrt(N), onde N = 2^n.
std::vector<std::vector<complexd>> qft_dagger(int n) {
    int dim = 1 << n;
    std::vector<std::vector<complexd>> Q(dim, std::vector<complexd>(dim));
    double N = static_cast<double>(dim);
    for (int i = 0; i < dim; ++i)
        for (int j = 0; j < dim; ++j)
            Q[i][j] = std::exp(complexd(0, -2.0 * M_PI * i * j / N)) / std::sqrt(N);
    return Q;
}

// Gera a matriz unitária de exponenciação modular para n qubits.
// Para cada base x, mapeia x → a^x mod N. Matriz de dimensão 2^n x 2^n.
std::vector<std::vector<complexd>> mod_exp_matrix(int a, int N, int n) {
    int dim = 1 << n;
    std::vector<std::vector<complexd>> U(dim, std::vector<complexd>(dim, 0));
    for (int x = 0; x < dim; ++x) {
        int y = 1;
        for (int i = 0; i < x; ++i) y = (y * a) % N;
        U[y % dim][x] = 1.0;
    }
    return U;
}

// Aplica uma matriz unitária a um vetor de estado usando OpenMP para paralelismo.
// A multiplicação matriz-vetor é feita linha por linha em paralelo.
// Número de threads definido por num_threads.
void apply_matrix_openmp(const std::vector<std::vector<complexd>> &U,
                         const std::vector<complexd> &state_in,
                         std::vector<complexd> &state_out,
                         int num_threads) {
    int N = U.size();
    state_out.resize(N, complexd(0, 0));
    #pragma omp parallel for num_threads(num_threads)
    for (int i = 0; i < N; ++i) {
        complexd acc = 0;
        for (int j = 0; j < N; ++j) {
            acc += U[i][j] * state_in[j];
        }
        state_out[i] = acc;
    }
}

// Executa a parte quântica do algoritmo de Shor simulada.
// Cria o estado inicial, aplica Hadamard, exponenciação modular, e QFT inversa.
// Mede o estado resultante com base nas probabilidades finais.
// Retorna o valor medido e tempos de execução de cada etapa.
QuantumResult run_quantum_shor(int a, int N, int n, int num_threads) {
    int dim = 1 << n;
    std::vector<complexd> state(dim, 0);
    state[1] = 1;

    auto start_total = std::chrono::high_resolution_clock::now();

    auto H = hadamard_all(n);
    auto Umod = mod_exp_matrix(a, N, n);
    auto QFTinv = qft_dagger(n);

    std::vector<complexd> temp1, temp2;

    auto start = std::chrono::high_resolution_clock::now();
    apply_matrix_openmp(H, state, temp1, num_threads);
    auto end = std::chrono::high_resolution_clock::now();
    double time_hadamard = std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();

    start = std::chrono::high_resolution_clock::now();
    apply_matrix_openmp(Umod, temp1, temp2, num_threads);
    end = std::chrono::high_resolution_clock::now();
    double time_modexp = std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();

    start = std::chrono::high_resolution_clock::now();
    apply_matrix_openmp(QFTinv, temp2, state, num_threads);
    end = std::chrono::high_resolution_clock::now();
    double time_qft = std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();

    std::vector<double> probs(dim);
    double total = 0.0;
    for (int i = 0; i < dim; ++i) {
        probs[i] = std::norm(state[i]);
        total += probs[i];
    }
    for (int i = 0; i < dim; ++i) probs[i] /= total;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::discrete_distribution<> d(probs.begin(), probs.end());

    int result = d(gen);

    auto end_total = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(end_total - start_total).count();

    return { result, elapsed, time_hadamard, time_modexp, time_qft };
}
