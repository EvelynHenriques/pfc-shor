#include "shor_quantum.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <algorithm>
#include <atomic>
#include <omp.h>
#include <sstream>
#include <random>
#include <sys/time.h>

static inline long long gcd(long long a, long long b) {
    while (b) { long long t = b; b = a % b; a = t; }
    return std::llabs(a);
}

static inline long long powmod(long long b, long long e, long long m) {
    long long r = 1 % m;
    while (e) {
        if (e & 1) r = (__int128)r * b % m;
        b = (__int128)b * b % m;
        e >>= 1;
    }
    return r;
}

static int contfrac(double x, int N) {
    long long h1 = 1, h0 = 0, k1 = 0, k0 = 1;
    long long best_k = -1;
    double best_err = 1e9;

    int iter = 0;
    while (iter++ < 1000) {
        long long a = floor(x);
        long long h2 = a * h1 + h0, k2 = a * k1 + k0;

        if (k2 > N) break;

        double approx = double(h2) / k2;
        double err = std::fabs(x - approx);
        if (err < best_err) {
            best_err = err;
            best_k = k2;
        }

        if (err < 10.0 / (2 * N)) return k2;

        h0 = h1; h1 = h2; k0 = k1; k1 = k2;
        double denom = x - a;
        if (std::fabs(denom) < 1e-12) break;
        x = 1.0 / denom;
    }

    if (best_k > 1 && best_k <= N) return best_k;
    return -1;
}

static std::vector<int> coprimes_uniform(int N, int lim, int num_bins = 20) {
    std::vector<std::vector<int>> faixas(num_bins);
    int intervalo = std::max(1, (N - 3) / num_bins);

    for (int a = 2; a < N - 1; ++a) {
        if (gcd(a, N) == 1) {
            int faixa = (a - 2) / intervalo;
            if (faixa >= num_bins) faixa = num_bins - 1;
            faixas[faixa].push_back(a);
        }
    }

    std::vector<int> result;
    std::mt19937 g(static_cast<unsigned>(time(NULL)));

    for (auto& bin : faixas) {
        if (!bin.empty()) {
            std::shuffle(bin.begin(), bin.end(), g);
            result.push_back(bin.front());
        }
    }

    std::vector<int> todos;
    for (const auto& bin : faixas)
        todos.insert(todos.end(), bin.begin(), bin.end());

    std::shuffle(todos.begin(), todos.end(), g);
    for (int a : todos) {
        if ((int)result.size() >= lim) break;
        if (std::find(result.begin(), result.end(), a) == result.end())
            result.push_back(a);
    }

    return result;
}

double seconds_between(const timeval& start, const timeval& end) {
    return (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1e6;
}

int main(int argc, char** argv) {
    setvbuf(stdout, NULL, _IONBF, 0);
    setvbuf(stderr, NULL, _IONBF, 0);
    std::cout.rdbuf(std::cerr.rdbuf());

    if (argc < 4) {
        std::cerr << "[ERRO] uso: " << argv[0] << " N total_threads internal_threads\n";
        return 1;
    }

    int N     = atoi(argv[1]);
    int TOTAL = atoi(argv[2]);
    int K     = atoi(argv[3]);
    int A     = std::max(1, TOTAL / K);
    int n     = ceil(log2(N));

    std::cerr << ">>> [DEBUG] Argumentos: N=" << N << ", TOTAL=" << TOTAL
              << ", INTERNAL=" << K << ", A=" << A << ", n=" << n << std::endl;

    omp_set_nested(1);
    omp_set_max_active_levels(2);
    omp_set_num_threads(A);

    srand(time(NULL));
    std::cerr << ">>> [DEBUG] Gerando coprimos..." << std::endl;
    auto As = coprimes_uniform(N, std::min(500, A * 10));
    std::cerr << ">>> [DEBUG] Coprimos gerados: " << As.size() << std::endl;

    std::cerr << ">>> [DEBUG] Lista completa de coprimos utilizados (As):" << std::endl;
    for (int a : As) std::cerr << a << " ";
    std::cerr << std::endl;

    std::atomic<long long> shots(0), succ(0), first_shot(-1);
    std::atomic<float> t_first_sec(-1.0f);
    long long P = 0, Q = 0;

    timeval t_global_start, t_global_end, t_wall_start;
    gettimeofday(&t_global_start, nullptr);
    gettimeofday(&t_wall_start, nullptr);

    #pragma omp parallel
    {
        #pragma omp single
        for (int a : As) {
            #pragma omp task firstprivate(a)
            {
                for (int i = 0; i < 100; ++i) {
                    long long id = shots.fetch_add(1);

                    if (id % 100 == 0) {
                        timeval now;
                        gettimeofday(&now, nullptr);
                        float elapsed = seconds_between(t_wall_start, now);
                        float est_total = (elapsed / (id + 1)) * (As.size() * 100);
                        float remaining = est_total - elapsed;

                        std::ostringstream oss;
                        oss << "[thread " << omp_get_thread_num()
                            << "] shot #" << id
                            << " | elapsed=" << elapsed << "s"
                            << " | est remaining=" << remaining << "s\n";
                        std::cerr << oss.str();
                    }

                    QuantumResult qr = run_quantum_shor(a, N, n, K);
                    double phase = double(qr.measurement) / double(1ULL << (2 * n));
                    int r = contfrac(phase, N);

                    // ?? DEBUG COMPLETO
                    #pragma omp critical
                    {
                        std::cerr << "[DEBUG] a=" << a
                                  << " | med=" << qr.measurement
                                  << " | phase=" << phase
                                  << " | r=" << r << std::endl;
                    }

                    if (r <= 1 || r % 2) continue;

                    long long x = powmod(a, r / 2, N), p = 0;
                    long long g1 = gcd(x - 1, N), g2 = gcd(x + 1, N);
                    if (g1 != 1 && g1 != N) p = g1;
                    else if (g2 != 1 && g2 != N) p = g2;

                    if (p) {
                        P = p;
                        Q = N / p;
                        succ++;

                        std::ostringstream oss;
                        oss << "[SUCESSO] a=" << a
                            << " fatores: " << P << " x " << Q
                            << " (shot " << id << ")\n";
                        std::cerr << oss.str();

                        long long exp = -1;
                        if (first_shot.compare_exchange_strong(exp, id)) {
                            timeval now;
                            gettimeofday(&now, nullptr);
                            float t_first = seconds_between(t_global_start, now);
                            t_first_sec.store(t_first);
                        }
                    }
                }
            }
        }
        #pragma omp taskwait
    }

    gettimeofday(&t_global_end, nullptr);
    float t_total = seconds_between(t_global_start, t_global_end);

    std::cerr << ">>> Resultado final:\n";
    std::cerr << "    N=" << N
              << "  fatores=" << P << " x " << Q
              << "  shots=" << shots
              << "  primeiro_shot=" << first_shot
              << "  t_first=" << t_first_sec.load() << "s"
              << "  t_total=" << t_total << "s"
              << "  sucessos=" << succ << std::endl;

    return 0;
}
