/* shor_quantum.cpp -- kernels compatíveis com GCC 4.8 / OpenMP 3.1 */
#include "shor_quantum.h"
#include <vector>
#include <complex>
#include <random>
#include <chrono>
#include <cmath>
#include <omp.h>

// 1) Usa float em vez de double para amplitudes
using cd = std::complex<float>;
const float INV_SQRT2 = 0.7071067811865475244f;

/* utilidades -------------------------------------------------*/
static inline long long gcd(long long a,long long b){
    while(b){ long long t=b; b=a%b; a=t; }
    return std::llabs(a);
}
static inline long long powmod(long long b,long long e,long long m){
    long long r = 1 % m;
    while(e){
        if(e & 1) r = (__int128)r * b % m;
        b = (__int128)b * b % m;
        e >>= 1;
    }
    return r;
}
static int contfrac(double x,int N){
    long long h1=1,h0=0,k1=0,k0=1;
    for(;;){
        long long a = floor(x);
        long long h2=a*h1+h0, k2=a*k1+k0;
        if(k2 > N) break;
        if(std::fabs(x - double(h2)/k2) < 1.0/(2*N)) return k2;
        h0=h1; h1=h2; k0=k1; k1=k2; x=1.0/(x-a);
    }
    return k1;
}

/* kernels 1-qubit / permuta ---------------------------------*/
static void hadamard(std::vector<cd>& v,int qb){
    size_t step = 1ULL << (qb+1);
    #pragma omp parallel for schedule(static)
    for(size_t base=0;base<v.size(); base+=step)
        for(size_t i=0;i<step/2;++i){
            cd a = v[base+i], b = v[base+i+step/2];
            v[base+i]         = (a + b) * INV_SQRT2;
            v[base+i+step/2]  = (a - b) * INV_SQRT2;
        }
}

static void cmodexp(std::vector<cd>& v,int ctrl,long long mult,long long N,int qctrl,int qtarg){
    size_t mctrl = 1ULL << ctrl, mtarg = (1ULL<<qtarg)-1;
    #pragma omp parallel for schedule(static)
    for(size_t idx=0; idx<v.size(); ++idx){
        if(!(idx & mctrl)) continue;
        size_t x = (idx >> qctrl) & mtarg;
        size_t y = (mult * x) % N;
        size_t dst = (idx & ((1ULL<<qctrl)-1)) | (y << qctrl);
        if(dst > idx) std::swap(v[idx], v[dst]);
    }
}

static void qft(std::vector<cd>& v,int m){
    for(int q=0;q<m;q++) hadamard(v,q);
    for(int d=1;d<m;d++){
        float ang = float(M_PI) / (1ULL<<d);
        cd phase = std::exp(cd(0,ang));
        for(int ctrl=0;ctrl<m-d;ctrl++){
            cd accum = 1.0f;
            size_t step = 1ULL << (ctrl + d + 1);
            #pragma omp parallel for schedule(static)
            for(size_t base=0;base<v.size(); base+=step){
                cd loc = accum;  // evita race
                for(size_t i=0;i<step/2;i++){
                    size_t idx = base + i + (1ULL<<ctrl);
                    if(idx & (1ULL<<ctrl)) v[idx] *= loc;
                }
            }
            accum *= phase;
        }
    }
}

/* circuito completo ----------------------------------------*/
QuantumResult run_quantum_shor(int a,int N,int n,int kthr){
    omp_set_num_threads(kthr);
    int qctrl = 2*n, qtarg = n, q = qctrl + qtarg;
    size_t dim = 1ULL << q;

    // 2) vetor de estado agora ocupa metade do espaço
    std::vector<cd> psi(dim, {0.0f, 0.0f});
    psi[1ULL << qctrl] = {1.0f, 0.0f};

    auto t0 = std::chrono::high_resolution_clock::now();
    for(int k=0;k<qctrl;k++) hadamard(psi,k);
    auto t1 = std::chrono::high_resolution_clock::now();

    std::vector<long long> apow(qctrl);
    apow[0] = a % N;
    for(int k=1;k<qctrl;k++)
        apow[k] = (__int128)apow[k-1] * apow[k-1] % N;
    for(int k=0;k<qctrl;k++)
        cmodexp(psi, k, apow[k], N, qctrl, qtarg);
    auto t2 = std::chrono::high_resolution_clock::now();

    qft(psi, qctrl);
    auto t3 = std::chrono::high_resolution_clock::now();

    size_t dimc = 1ULL << qctrl;
    // 3) probabilidade também em float
    std::vector<float> prob(dimc, 0.0f);

    #pragma omp parallel
    {
        std::vector<float> loc(dimc, 0.0f);
        #pragma omp for nowait
        for(size_t i=0;i<psi.size();++i)
            loc[i & (dimc-1)] += std::norm(psi[i]);
        #pragma omp critical
        for(size_t i=0;i<dimc;++i)
            prob[i] += loc[i];
    }

    std::random_device rd;
    std::mt19937_64 gen(rd());
    std::discrete_distribution<size_t> dist(prob.begin(), prob.end());
    int m = dist(gen);

    double th = std::chrono::duration<double>(t1-t0).count();
    double tm = std::chrono::duration<double>(t2-t1).count();
    double tq = std::chrono::duration<double>(t3-t2).count();
    double wall = std::chrono::duration<double>(t3-t0).count();

    return { m, wall, th, tm, tq };
}
