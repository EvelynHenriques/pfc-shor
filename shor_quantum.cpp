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

// Funçção para gerar a matriz Hadamard para n qubits
// A matriz Hadamard é uma matriz quadrada de tamanho 2^n x 2^n
// onde cada elemento é 1/sqrt(2^n) multiplicado por (-1)^k
// onde k é o número de bits 1 na representação binária do índice i & j.
// A matriz Hadamard é usada para criar superposições quânticas.
// A matriz Hadamard é uma matriz unitária que transforma um vetor de estado
// em uma superposição de estados.
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

// Função para gerar a matriz QFT (Transformada de Fourier Quântica)
// A matriz QFT é uma matriz unitária que transforma um vetor de estado
// em uma superposição de estados com base na Transformada de Fourier.
// A matriz QFT é usada para medir a fase de um vetor de estado.
// A matriz QFT é uma matriz quadrada de tamanho 2^n x 2^n
// onde cada elemento é exp(2 * pi * i * k / N) / sqrt(N)
// onde k é o índice da linha e j é o índice da coluna.
std::vector<std::vector<complexd>> qft_dagger(int n) {
    int dim = 1 << n;
    std::vector<std::vector<complexd>> Q(dim, std::vector<complexd>(dim));
    double N = static_cast<double>(dim);
    for (int i = 0; i < dim; ++i)
        for (int j = 0; j < dim; ++j)
            Q[i][j] = std::exp(complexd(0, -2.0 * M_PI * i * j / N)) / std::sqrt(N);
    return Q;
}

// Função para gerar a matriz de exponenciação modular
// A matriz de exponenciação modular é uma matriz unitária que transforma
// um vetor de estado em uma superposição de estados com base na exponenciação modular.
// A matriz de exponenciação modular é usada para calcular a exponenciação modular
// de um número a com base em um número N.
// A matriz de exponenciação modular é uma matriz quadrada de tamanho 2^n x 2^n
// onde cada elemento é 1 se o índice j é igual a (a^i mod N) e 0 caso contrário.
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

// Função para aplicar uma matriz a um vetor de estado
// A função aplica uma matriz unitária a um vetor de estado
// e retorna o vetor de estado resultante.
// A função usa OpenMP para paralelizar a multiplicação da matriz
// com o vetor de estado. O número de threads é especificado
// pelo parâmetro num_threads.
// A função usa a diretiva #pragma omp parallel for para paralelizar
// o loop que aplica a matriz ao vetor de estado.
// A função usa a diretiva #pragma omp critical para garantir
// que o acesso ao vetor de estado resultante é seguro entre threads.
// A função usa a diretiva #pragma omp atomic para garantir
// que a atualização do vetor de estado resultante é atômica
// entre threads.
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

// Função principal que executa o algoritmo de Shor
// A função recebe um número inteiro a, um número inteiro N,
// o número de qubits n e o número de threads num_threads.
// A função retorna um objeto QuantumResult que contém
// o resultado da medição e o tempo gasto para executar
// o algoritmo.
QuantumResult run_quantum_shor(int a, int N, int n, int num_threads) {
    int dim = 1 << n;
    std::vector<complexd> state(dim, 0);
    state[1] = 1;

    auto H = hadamard_all(n);
    auto Umod = mod_exp_matrix(a, N, n);
    auto QFTinv = qft_dagger(n);

    std::vector<complexd> temp1, temp2;

    auto start = std::chrono::high_resolution_clock::now();
    apply_matrix_openmp(H, state, temp1, num_threads);
    apply_matrix_openmp(Umod, temp1, temp2, num_threads);
    apply_matrix_openmp(QFTinv, temp2, state, num_threads);
    auto end = std::chrono::high_resolution_clock::now();

    double elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();

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
    return { result, elapsed };
}
