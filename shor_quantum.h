#ifndef SHOR_QUANTUM_H
#define SHOR_QUANTUM_H

#include <vector>
#include <complex>

typedef std::complex<double> complexd;

struct QuantumResult {
    int measurement;
    double elapsed_time;
};

QuantumResult run_quantum_shor(int a, int N, int n, int num_threads);

#endif
