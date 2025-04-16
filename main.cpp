#include "shor_quantum.h"
#include <iostream>
#include <fstream>
#include <bitset>
#include <numeric>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <atomic>
#include <omp.h>
#include <mutex>
#include <queue>
#include <set>
#include <vector>
#include <algorithm>
#include <random>

std::vector<int> generate_random_coprimes(int N, int count) {
    std::vector<int> coprimes;
    for (int i = 2; i <= N - 2; ++i) {
        if (std::gcd(i, N) == 1) {
            coprimes.push_back(i);
        }
    }

    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(coprimes.begin(), coprimes.end(), g);

    if ((int)coprimes.size() > count) {
        coprimes.resize(count);
    }
    return coprimes;
}

int main(int argc, char* argv[]) {
    std::cout << "Iniciando programa Shor com OpenMP...\n";
    if (argc < 3) {
        std::cerr << "Uso: " << argv[0] << " <N> <max_threads>" << std::endl;
        return 1;
    }

    int N = std::atoi(argv[1]);
    int MAX_THREADS = std::atoi(argv[2]);
    int n = static_cast<int>(std::ceil(std::log2(N)));
    int NUM_THREADS_EXTERNOS = std::max(1, MAX_THREADS / 2);
    int NUM_THREADS_INTERNOS = std::max(1, MAX_THREADS / NUM_THREADS_EXTERNOS);
    int TENTATIVAS_POR_A = 15;
    int NUM_A = std::min(NUM_THREADS_EXTERNOS * 10, 500);

    std::ofstream csv("shor_resultados.csv");
    csv << "thread,a,tentativa,medicao,binario,fase,r,x,f1,f2,tempo,success\n";

    std::vector<int> valores_a = generate_random_coprimes(N, NUM_A);
    std::cout << "Gerados " << valores_a.size() << " valores a coprimos com N = " << N << "\n";

    std::atomic<bool> found(false);
    std::mutex csv_mutex;

    std::cout << "Iniciando paralelismo com " << NUM_THREADS_EXTERNOS 
              << " threads externas e " << NUM_THREADS_INTERNOS 
              << " internas por execucao...\n";

    #pragma omp parallel
    {
        #pragma omp single
        {
            std::cout << "Numero de threads ativos: "
                      << omp_get_num_threads() << std::endl;
        }
    }

    #pragma omp parallel for num_threads(NUM_THREADS_EXTERNOS)
    for (int i = 0; i < static_cast<int>(valores_a.size()); ++i) {
        if (found) continue;

        int tid = omp_get_thread_num();
        int a = valores_a[i];

        #pragma omp critical
        std::cout << "[Thread " << tid << "] Testando a = " << a << std::endl;

        for (int tentativa = 0; tentativa < TENTATIVAS_POR_A && !found; ++tentativa) {
            QuantumResult qr = run_quantum_shor(a, N, n, NUM_THREADS_INTERNOS);

            int dim = 1 << n;
            double phase = static_cast<double>(qr.measurement) / dim;

            int best_r = 0;
            double best_diff = 1.0;
            for (int denom = 1; denom <= N; ++denom) {
                int num = static_cast<int>(std::round(phase * denom));
                double approx = static_cast<double>(num) / denom;
                double diff = std::abs(phase - approx);
                if (diff < best_diff) {
                    best_diff = diff;
                    best_r = denom;
                }
            }

            int x = static_cast<int>(std::pow(a, best_r / 2)) % N;
            int f1 = std::gcd(x - 1, N);
            int f2 = std::gcd(x + 1, N);

            bool success = false;
            if (best_r % 2 == 0 && x != 1 && x != N - 1 && f1 * f2 == N && f1 != 1 && f2 != 1) {
                success = true;
                found = true;
            }

            {
                std::lock_guard<std::mutex> lock(csv_mutex);
                csv << tid << "," << a << "," << (tentativa + 1) << "," << qr.measurement << ","
                    << std::bitset<8>(qr.measurement) << "," << phase << ","
                    << best_r << "," << x << "," << f1 << "," << f2 << ","
                    << qr.elapsed_time << "," << (success ? "1" : "0") << "\n";
            }

            if (success) {
                std::cout << "\nThread " << tid << " encontrou fatores: " << f1 << " e " << f2 << std::endl;
                break;
            }
        }
    }

    csv.close();

    if (!found) {
        std::cout << "\nNenhum fator encontrado." << std::endl;
        return 1;
    }

    return 0;
}
