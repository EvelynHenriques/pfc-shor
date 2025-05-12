#include "shor_quantum.h"
#include <iostream>
#include <bitset>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <atomic>
#include <omp.h>
#include <vector>
#include <algorithm>
#include <random>
#include <chrono>

int gcd(int a, int b) {
    while (b != 0) {
        int t = b;
        b = a % b;
        a = t;
    }
    return std::abs(a);
}

/* --------------------------------------------------------------------------
 * Gera 'count' valores de a coprimos com N, distribuídos uniformemente
 * pelo intervalo [2, N-2] (amostragem estratificada).
 * -------------------------------------------------------------------------- */
std::vector<int> generate_stratified_coprimes(int N, int count) {
    std::vector<int> todos;
    for (int i = 2; i <= N - 2; ++i) {
        if (gcd(i, N) == 1) {      
            todos.push_back(i);
        }
    }

    // Se houver poucos coprimos, basta embaralhar e retornar
    if (static_cast<int>(todos.size()) <= count) {
        std::mt19937 rng(static_cast<unsigned>(std::time(nullptr)));
        std::shuffle(todos.begin(), todos.end(), rng);
        return todos;
    }

    /* --- Estratificação --- */
    std::vector<std::vector<int> > faixas(count);
    for (int a : todos) {
        int idx = ((a - 2) * count) / (N - 3);   // faixa numérica
        if (idx >= count) idx = count - 1;       // proteção
        faixas[idx].push_back(a);
    }

    std::vector<int> selecionados;
    std::mt19937 rng(static_cast<unsigned>(std::time(nullptr)));

    // Sorteia 1 coprimo de cada faixa não-vazia
    for (size_t i = 0; i < faixas.size(); ++i) {
        if (!faixas[i].empty()) {
            std::shuffle(faixas[i].begin(), faixas[i].end(), rng);
            selecionados.push_back(faixas[i].front());
        }
    }

    // Se faltou preencher (faixas vazias), completa aleatoriamente
    if (static_cast<int>(selecionados.size()) < count) {
        std::shuffle(todos.begin(), todos.end(), rng);
        for (int a : todos) {
            if (static_cast<int>(selecionados.size()) >= count) break;
            if (std::find(selecionados.begin(), selecionados.end(), a) == selecionados.end()) {
                selecionados.push_back(a);
            }
        }
    }
    return selecionados;
}

int main(int argc, char* argv[]) {
    std::cout << "Iniciando programa Shor com OpenMP...\n";

    /* ------------------- leitura de parâmetros ------------------- */
    if (argc < 4) {
        std::cerr << "Uso: " << argv[0]
                  << " <integer_to_factor> <num_threads> <internal_threads>\n";
        return 1;
    }
    int N                  = std::atoi(argv[1]);
    int TOTAL_THREADS      = std::atoi(argv[2]);
    int THREADS_INTERNOS   = std::atoi(argv[3]);

    if (N < 3 || TOTAL_THREADS < 1 || THREADS_INTERNOS < 1) {
        std::cerr << "Parâmetros inválidos.\n";
        return 1;
    }

    int THREADS_EXTERNOS = std::max(1, TOTAL_THREADS / THREADS_INTERNOS);
    int n                = static_cast<int>(std::ceil(std::log2(N)));
    int TENTATIVAS_POR_A = 100;
    int NUM_A            = std::min(THREADS_EXTERNOS * 10, 500);

    std::cout << "Número a fatorar: " << N << "\n";
    std::cout << "Threads: " << TOTAL_THREADS << "  →  "
              << THREADS_EXTERNOS << " externas + "
              << THREADS_INTERNOS << " internas\n";

    std::vector<int> valores_a = generate_stratified_coprimes(N, NUM_A);
    std::cout << "Coprimos gerados: " << valores_a.size() << "\n";

    std::atomic<bool> found(false);   // sinal global de parada

    /* Log rápido: quantas threads OpenMP ativas neste ponto */
    #pragma omp parallel
    #pragma omp single
    std::cout << "Threads OpenMP ativas: " << omp_get_num_threads() << "\n";

    /* ------------------------ loop paralelo ---------------------- */
    #pragma omp parallel for schedule(dynamic, 1) num_threads(THREADS_EXTERNOS)
    for (int idx = 0; idx < static_cast<int>(valores_a.size()); ++idx) {
        if (found.load()) continue;

        int tid = omp_get_thread_num();
        int a   = valores_a[idx];

        int tentativas_realizadas = 0;
        bool success = false;
        int f1 = 0, f2 = 0;
        QuantumResult best_qr = {};
        int best_r = 0, best_x = 0;

        auto t_start = std::chrono::high_resolution_clock::now();

        for (int t = 0; t < TENTATIVAS_POR_A; ++t) {
            if (found.load()) break;

            QuantumResult qr = run_quantum_shor(a, N, n, THREADS_INTERNOS);
            ++tentativas_realizadas;

            /* ---------- estima r a partir da fase medida ---------- */
            int dim      = 1 << n;
            double phase = static_cast<double>(qr.measurement) / dim;

            int r_est = 0;
            double best_diff = 1.0;
            for (int denom = 1; denom <= N; ++denom) {
                int num   = static_cast<int>(std::round(phase * denom));
                double diff = std::fabs(phase - static_cast<double>(num) / denom);
                if (diff < best_diff) {
                    best_diff = diff;
                    r_est = denom;
                }
            }

            int x  = static_cast<int>(std::pow(a, r_est / 2)) % N;
            int g1 = gcd(x - 1, N);
            int g2 = gcd(x + 1, N);

            best_qr = qr;
            best_r  = r_est;
            best_x  = x;

            /* ---------- condição de sucesso de Shor --------------- */
            if (r_est % 2 == 0 && x != 1 && x != N - 1 &&
                g1 * g2 == N && g1 != 1 && g2 != 1) {
                success = true;
                f1 = g1;  f2 = g2;
                found.store(true);
                break;
            }
        }

        double tempo_total =
            std::chrono::duration_cast<std::chrono::duration<double>>
            (std::chrono::high_resolution_clock::now() - t_start).count();

        /* ----------------------- log por thread ------------------- */
        std::cout << "----------------------------------------------------\n";
        std::cout << "[Thread " << tid << "]  a = " << a << "\n";
        std::cout << "Tentativas: " << tentativas_realizadas
                  << "   Tempo: " << tempo_total << " s\n";

        if (success) {
            std::cout << "Fatores: " << f1 << " × " << f2 << "\n";
            std::cout << "Measurement: " << best_qr.measurement
                      << " (bin " << std::bitset<16>(best_qr.measurement) << ")\n";
            std::cout << "r = " << best_r << "   x = " << best_x << "\n";
            std::cout << "Hadamard " << best_qr.time_hadamard
                      << "s | ModExp " << best_qr.time_modexp
                      << "s | QFT "    << best_qr.time_qft << "s\n";
        } else {
            std::cout << "Nenhum fator válido.\n";
        }
        std::cout << "----------------------------------------------------\n\n";
    }

    if (!found) {
        std::cout << "Nenhuma thread encontrou fatores.\n";
        return 1;
    }
    return 0;
}
