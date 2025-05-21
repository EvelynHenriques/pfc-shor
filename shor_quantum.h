#ifndef SHOR_QUANTUM_H
#define SHOR_QUANTUM_H
struct QuantumResult{
    int measurement;
    double wall_time;
    double time_hadamard;
    double time_modexp;
    double time_qft;
};
QuantumResult run_quantum_shor(int a,int N,int n,int k_threads);
#endif
