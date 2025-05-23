#include "shor_quantum.h"
#include <vector>
#include <complex>
#include <random>
#include <chrono>
#include <cmath>
#include <omp.h>

using cd = std::complex<float>;
const float INV_SQRT2 = 0.7071067811865475244f;

/* Utilidades matem�ticas */
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

/* Hadamard no qubit qb */
static void hadamard(std::vector<cd>& v, int qb) {
    size_t step = 1ULL << (qb + 1);
    #pragma omp parallel for schedule(static)
    for (size_t base = 0; base < v.size(); base += step)
        for (size_t i = 0; i < step / 2; ++i) {
            cd a = v[base + i], b = v[base + i + step / 2];
            v[base + i] = (a + b) * INV_SQRT2;
            v[base + i + step / 2] = (a - b) * INV_SQRT2;
        }
}

/* Controlled modular exponentiation */
static void cmodexp(std::vector<cd>& v, int ctrl, long long mult, long long N, int qctrl, int qtarg) {
    size_t mctrl = 1ULL << ctrl;
    size_t mtarg = (1ULL << qtarg) - 1;

    std::vector<cd> temp = v;
    #pragma omp parallel for
    for (size_t idx = 0; idx < v.size(); ++idx) {
        if (!(idx & mctrl)) continue;
        size_t x = (idx >> qctrl) & mtarg;
        size_t y = (mult * x) % N;
        size_t dst = (idx & ((1ULL << qctrl) - 1)) | (y << qctrl);
        v[dst] = temp[idx];
    }
}

/* QFT correta nos qctrl bits */
static void qft(std::vector<cd>& v, int m) {
    size_t dim = 1ULL << m;

    // Aplica Hadamard e controlled-phase
    for (int i = 0; i < m; ++i) {
        size_t shift = 1ULL << i;
        for (size_t j = 0; j < dim; ++j) {
            if ((j & shift) == 0) {
                size_t partner = j | shift;
                cd a = v[j];
                cd b = v[partner];
                v[j] = (a + b) * INV_SQRT2;
                v[partner] = (a - b) * INV_SQRT2;
            }
        }

        for (int k = 2; k <= m - i; ++k) {
            double angle = 2.0 * M_PI / (1ULL << k);
            for (size_t j = 0; j < dim; ++j) {
                if (((j >> i) & 1) && ((j >> (i + k - 1)) & 1)) {
                    v[j] *= std::exp(cd(0, angle));
                }
            }
        }
    }

    // Bit reversal
    for (size_t i = 0; i < dim; ++i) {
        size_t rev = 0, x = i;
        for (int j = 0; j < m; ++j) {
            rev = (rev << 1) | (x & 1);
            x >>= 1;
        }
        if (i < rev) std::swap(v[i], v[rev]);
    }
}

/* Fun��o principal de simula��o */
QuantumResult run_quantum_shor(int a, int N, int n, int kthr) {
    omp_set_num_threads(kthr);
    int qctrl = 2 * n, qtarg = n, q = qctrl + qtarg;
    size_t dim = 1ULL << q;

    std::vector<cd> psi(dim, {0.0f, 0.0f});
    psi[0] = {1.0f, 0.0f};  // Correto: estado |0?

    auto t0 = std::chrono::high_resolution_clock::now();

    for (int k = 0; k < qctrl; ++k)
        hadamard(psi, k);
    auto t1 = std::chrono::high_resolution_clock::now();

    std::vector<long long> apow(qctrl);
    apow[0] = a % N;
    for (int k = 1; k < qctrl; ++k)
        apow[k] = (__int128)apow[k - 1] * apow[k - 1] % N;

    for (int k = 0; k < qctrl; ++k)
        cmodexp(psi, k, apow[k], N, qctrl, qtarg);
    auto t2 = std::chrono::high_resolution_clock::now();

    qft(psi, qctrl);
    auto t3 = std::chrono::high_resolution_clock::now();

    size_t dimc = 1ULL << qctrl;
    std::vector<float> prob(dimc, 0.0f);

    #pragma omp parallel
    {
        std::vector<float> loc(dimc, 0.0f);
        #pragma omp for nowait
        for (size_t i = 0; i < psi.size(); ++i)
            loc[i & (dimc - 1)] += std::norm(psi[i]);
        #pragma omp critical
        for (size_t i = 0; i < dimc; ++i)
            prob[i] += loc[i];
    }

    std::random_device rd;
    std::mt19937_64 gen(rd());
    std::discrete_distribution<size_t> dist(prob.begin(), prob.end());
    int m = dist(gen);

    double th = std::chrono::duration<double>(t1 - t0).count();
    double tm = std::chrono::duration<double>(t2 - t1).count();
    double tq = std::chrono::duration<double>(t3 - t2).count();
    double wall = std::chrono::duration<double>(t3 - t0).count();

    return { m, wall, th, tm, tq };
}
