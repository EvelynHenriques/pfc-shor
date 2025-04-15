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

// Gera um número aleatório a entre 2 e N-2 que seja coprimo de N
int generate_random_coprime(int N, int seed_offset = 0) {
    std::srand(static_cast<unsigned int>(std::time(nullptr)) + seed_offset);
    while (true) {
        int a = 2 + std::rand() % (N - 3);
        if (std::gcd(a, N) == 1)
            return a;
    }
}


int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Uso: " << argv[0] << " <N>" << std::endl;
        return 1;
    }

    int N = std::atoi(argv[1]); // Número a ser fatorado
    int n = static_cast<int>(std::ceil(std::log2(N)));  // Número de qubits necessários
    int num_threads_inner = 4;  // Número de threads para a execução interna do algoritmo
    const int TENTATIVAS_POR_A = 5;  // Número de tentativas para cada valor de a
    const int NUM_A = 8;    // Número de valores de a a serem testados

    std::ofstream csv("shor_resultados.csv");
    csv << "thread,a,tentativa,medicao,binario,fase,r,x,f1,f2,tempo,success\n";

    std::atomic<bool> found(false);
    std::mutex mtx;

    #pragma omp parallel for num_threads(NUM_A)
    // Inicia um loop paralelo com OpenMP. Cada iteração (thread) recebe um valor único de 'tid'.
    // Cada thread testará um valor diferente de 'a'.
    for (int tid = 0; tid < NUM_A; ++tid) {

        // Verifica se outra thread já encontrou fatores (variável atômica).
        // Se sim, essa thread pula sua execução.
        if (found) continue;

        // Gera um valor aleatório de 'a' que seja coprimo de N.
        // Usa um offset baseado em tid para gerar seeds diferentes por thread.
        int a = generate_random_coprime(N, tid * 1000);
        std::cout << "Thread " << tid << " tentando a = " << a << std::endl;

        for (int i = 0; i < TENTATIVAS_POR_A && !found; ++i) {

            // Chama a simulação da parte quântica com o valor de 'a'.
            // Retorna o resultado da medição e o tempo gasto.
            QuantumResult qr = run_quantum_shor(a, N, n, num_threads_inner);

            // Converte a medição para um valor de fase: result / 2^n
            // O valor de fase é usado para encontrar o melhor denominador (best_r).
            // O melhor denominador é aquele que minimiza a diferença entre
            // a fase real e a fase aproximada (num / denom).
            // O melhor denominador é usado para calcular x, que é usado
            // para calcular f1 e f2, os fatores de N.
            int dim = 1 << n;
            double phase = static_cast<double>(qr.measurement) / dim;

            // Estima o período 'r' a partir da fase usando aproximação de fração
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
            
            // Calcula x = a^(r/2) mod N
            // e os fatores f1 e f2 a partir de x.
            // Se f1 e f2 forem válidos, marca como sucesso.
            // Caso contrário, continua tentando.
            int x = static_cast<int>(std::pow(a, best_r / 2)) % N;
            int f1 = std::gcd(x - 1, N);
            int f2 = std::gcd(x + 1, N);
            bool success = false;

            // Verifica se os fatores são válidos e não triviais
            // (diferentes de 1 e N).
            if (best_r % 2 == 0 && x != 1 && x != N - 1 && f1 * f2 == N && f1 != 1 && f2 != 1) {
                success = true;
                found = true;
            }

            // Se a thread encontrar fatores válidos, printa os resultados
            // e salva no arquivo CSV.
            // Se não, continua tentando.
            std::lock_guard<std::mutex> lock(mtx);
            csv << tid << "," << a << "," << (i+1) << "," << qr.measurement << ","
                << std::bitset<8>(qr.measurement) << "," << phase << ","
                << best_r << "," << x << "," << f1 << "," << f2 << ","
                << qr.elapsed_time << "," << (success ? "1" : "0") << "\n";

            if (success) {
                std::cout << "\nThread " << tid << " encontrou fatores: " << f1 << " e " << f2 << std::endl;
                break;
            }
        }
    }

    csv.close();

    if (!found) {
        std::cout << "\nNenhuma thread encontrou fatores validos." << std::endl;
        return 1;
    }

    return 0;
}
