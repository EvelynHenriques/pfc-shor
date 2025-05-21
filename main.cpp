#include "shor_quantum.h"
#include <bits/stdc++.h>
#include <atomic>
#include <omp.h>

/* --- utilidades auxiliares -----------------------------------------*/
static inline long long gcd(long long a,long long b){
    while(b){ long long t=b; b=a%b; a=t;} return std::llabs(a);
}
static inline long long powmod(long long b,long long e,long long m){
    long long r=1%m; while(e){ if(e&1) r=(__int128)r*b%m;
                               b=(__int128)b*b%m; e>>=1;} return r;
}
static int contfrac(double x,int N){
    long long h1=1,h0=0,k1=0,k0=1;
    while(true){
        long long a=floor(x);
        long long h2=a*h1+h0, k2=a*k1+k0;
        if(k2>N) break;
        if(std::fabs(x-double(h2)/k2) < 1.0/(2*N)) return k2;
        h0=h1; h1=h2; k0=k1; k1=k2; x=1.0/(x-a);
    }
    return k1;
}
static std::vector<int> coprimes(int N,int lim){
    std::vector<int> v; for(int a=2;a<N-1;++a) if(gcd(a,N)==1) v.push_back(a);
    std::shuffle(v.begin(),v.end(),std::mt19937((unsigned)time(nullptr)));
    if((int)v.size()>lim) v.resize(lim); return v;
}

/* ------------------------------------------------------------------*/
int main(int argc,char**argv){
    if(argc<4){ fprintf(stderr,"uso: %s N total_threads internal_threads\n",argv[0]); return 1;}
    int N     = atoi(argv[1]);
    int TOTAL = atoi(argv[2]);          /* threads nível externo   */
    int K     = atoi(argv[3]);          /* threads internas kernel */
    int A     = std::max(1,TOTAL/K);    /* tasks paralelas de 'a'  */
    int n     = ceil(log2(N));          /* bits de N               */

    omp_set_nested(1); omp_set_max_active_levels(2); omp_set_num_threads(A);

    auto As = coprimes(N,std::min(500,A*10));
    const int SHOTS = 100;

    std::atomic<long long> shots(0), succ(0), first_shot(-1);
    std::atomic<double>    t_first_sec(-1.0);

    long long P=0,Q=0;
    auto t_global_start = std::chrono::high_resolution_clock::now();

    /* ---------- nível 1: tasks externas (candidatos a) ------------ */
    #pragma omp parallel
    {
        #pragma omp single
        for(int a:As)
            #pragma omp task firstprivate(a)
            for(int s=0;s<SHOTS;++s){
                long long id = shots++;

                QuantumResult qr = run_quantum_shor(a,N,n,K);

                double phase = double(qr.measurement) / double(1ULL<<(2*n));
                int r = contfrac(phase,N); if(r%2) continue;

                long long x = powmod(a,r/2,N), p=0;
                long long g1=gcd(x-1,N), g2=gcd(x+1,N);
                if(g1!=1&&g1!=N) p=g1; else if(g2!=1&&g2!=N) p=g2;

                if(p){
                    P=p; Q=N/p; succ++;
                    long long exp=-1;
                    if(first_shot.compare_exchange_strong(exp,id)){
                        auto now=std::chrono::high_resolution_clock::now();
                        t_first_sec.store(
                          std::chrono::duration<double>(now-t_global_start).count());
                    }
                }
            }
        #pragma omp taskwait
    }

    auto t_global_end = std::chrono::high_resolution_clock::now();
    double t_total = std::chrono::duration<double>(t_global_end - t_global_start).count();

    std::cout << "N=" << N
              << "  fatores " << P << ' ' << Q
              << "  shots " << shots
              << "  primeiro_shot " << first_shot
              << "  t_first " << t_first_sec.load() << " s"
              << "  t_total " << t_total << " s"
              << "  sucessos " << succ << '\n';
    return 0;
}
