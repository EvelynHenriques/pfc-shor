#include "shor_quantum.h"
#include <iostream>
#include <fstream>
#include <bitset>
#include <numeric>
#include <cmath>
#include <cstdlib>
#include <ctime>

int generate_random_coprime(int N) {
    while (true) {
        int a = 2 + std::rand() % (N - 3); // a ∈ [2, N-2]
        if (std::gcd(a, N) == 1)
            return a;
    }
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Uso: " << argv[0] << " <N>" << std::endl;
        return 1;
    }

    int N = std::atoi(argv[1]);
    int num_threads = 4;
    const int MAX_TRIES = 100;
    const int TENTATIVAS_POR_A = 5;

    int n = static_cast<int>(std::ceil(std::log2(N)));

    std::ofstream csv("shor_resultados.csv");
    csv << "tentativa,a,medicao,binario,fase,r,tempo\n";

    std::srand(static_cast<unsigned int>(std::time(nullptr)));

    int attempts = 0;
    double total_time = 0.0;

    while (attempts < MAX_TRIES) {
        int a = generate_random_coprime(N);
        std::cout << "\nNovo a gerado: " << a << std::endl;

        for (int i = 0; i < TENTATIVAS_POR_A && attempts < MAX_TRIES; ++i) {
            attempts++;
            std::cout << "\nTentativa " << attempts << " com a = " << a << std::endl;

            QuantumResult qr = run_quantum_shor(a, N, n, num_threads);
            total_time += qr.elapsed_time;

            int dim = 1 << n;
            double phase = static_cast<double>(qr.measurement) / dim;

            const int MAX_DENOM = N;
            int best_r = 0;
            double best_diff = 1.0;
            for (int denom = 1; denom <= MAX_DENOM; ++denom) {
                int num = static_cast<int>(std::round(phase * denom));
                double approx = static_cast<double>(num) / denom;
                double diff = std::abs(phase - approx);
                if (diff < best_diff) {
                    best_diff = diff;
                    best_r = denom;
                }
            }

            csv << attempts << "," << a << "," << qr.measurement << "," 
                << std::bitset<8>(qr.measurement) << "," << phase << ","
                << best_r << "," << qr.elapsed_time << "\n";

            std::cout << "Tempo util (simulacao quantica): " << qr.elapsed_time << " segundos" << std::endl;
            std::cout << "Medicao: " << qr.measurement << " (binario: " << std::bitset<8>(qr.measurement) << ")" << std::endl;
            std::cout << "Fase estimada: " << phase << std::endl;
            std::cout << "Estimativa de r: " << best_r << std::endl;

            if (best_r % 2 != 0) {
                std::cout << "r impar, descartado." << std::endl;
                continue;
            }

            int x = static_cast<int>(std::pow(a, best_r / 2)) % N;
            if (x == 1 || x == N - 1) {
                std::cout << "x = +-1 mod N, descartado." << std::endl;
                continue;
            }

            int f1 = std::gcd(x - 1, N);
            int f2 = std::gcd(x + 1, N);

            if (f1 * f2 == N && f1 != 1 && f2 != 1) {
                std::cout << " Fatores encontrados: " << f1 << " e " << f2 << std::endl;
                std::cout << "Total de tentativas: " << attempts << std::endl;
                std::cout << "Tempo total util (parte quantica): " << total_time << " segundos" << std::endl;
                csv.close();
                return 0;
            } else {
                std::cout << "Fatores triviais encontrados ou erro." << std::endl;
            }
        }
    }

    std::cout << "\n Nenhum fator valido encontrado apos " << MAX_TRIES << " tentativas." << std::endl;
    std::cout << "Tempo total util (parte quantica): " << total_time << " segundos" << std::endl;
    csv.close();
    return 1;
}
