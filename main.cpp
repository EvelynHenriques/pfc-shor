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
#include <chrono>
#include <filesystem>
namespace fs = std::filesystem;

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
    csv << "thread,a,tentativa,medicao,binario,fase,r,x,f1,f2,tempo,tempo_hadamard,tempo_modexp,tempo_qft,success\n";

    std::vector<int> valores_a = generate_random_coprimes(N, NUM_A);
    std::cout << "Gerados " << valores_a.size() << " valores a coprimos com N = " << N << "\n";

    std::atomic<bool> found(false);
    std::mutex csv_mutex;

    // Cria pasta de logs temporários
    fs::create_directory("logs_tmp");

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

        auto t_start = std::chrono::high_resolution_clock::now();
        int tentativas_realizadas = 0;
        bool success = false;
        int f1 = 0, f2 = 0;

        for (int tentativa = 0; tentativa < TENTATIVAS_POR_A && !found; ++tentativa) {
            QuantumResult qr = run_quantum_shor(a, N, n, NUM_THREADS_INTERNOS);
            tentativas_realizadas++;

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
            int g1 = std::gcd(x - 1, N);
            int g2 = std::gcd(x + 1, N);

            if (best_r % 2 == 0 && x != 1 && x != N - 1 && g1 * g2 == N && g1 != 1 && g2 != 1) {
                success = true;
                found = true;
                f1 = g1;
                f2 = g2;
            }

            {
                std::lock_guard<std::mutex> lock(csv_mutex);
                csv << tid << "," << a << "," << (tentativa + 1) << "," << qr.measurement << ","
                    << std::bitset<8>(qr.measurement) << "," << phase << ","
                    << best_r << "," << x << "," << g1 << "," << g2 << ","
                    << qr.elapsed_time << ","
                    << qr.time_hadamard << "," << qr.time_modexp << "," << qr.time_qft << ","
                    << (success ? "1" : "0") << "\n";
            }

            if (success) break;
        }

        auto t_end = std::chrono::high_resolution_clock::now();
        double tempo_total = std::chrono::duration_cast<std::chrono::duration<double>>(t_end - t_start).count();

        auto timestamp = std::chrono::system_clock::now().time_since_epoch().count();
        std::string log_filename = "logs_tmp/log_" + std::to_string(tid) + "_" + std::to_string(timestamp) + ".txt";
        std::ofstream log_file(log_filename);

        log_file << "[Thread " << tid << "] Testou a = " << a
                 << " por " << tentativas_realizadas << " tentativa(s), "
                 << "levou " << tempo_total << "s. "
                 << (success ? "Fatores encontrados: " + std::to_string(f1) + " e " + std::to_string(f2) : "Sem sucesso.")
                 << "\n";

        log_file.close();
    }

    std::ofstream final_log("log_final.txt");
    for (const auto& entry : fs::directory_iterator("logs_tmp")) {
        std::ifstream temp_log(entry.path());
        final_log << temp_log.rdbuf();
        temp_log.close();
    }

    fs::remove_all("logs_tmp");
    csv.close();

    if (!found) {
        std::cout << "\nNenhum fator encontrado." << std::endl;
        return 1;
    }

    return 0;
}
