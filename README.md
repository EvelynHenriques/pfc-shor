# ⚛️ Shor Modular com Paralelismo por 'a'

Este projeto implementa o **algoritmo completo de Shor** com paralelização via **OpenMP**, utilizando **paralelismo externo por múltiplos valores de `a`**. A parte quântica do algoritmo é simulada com multiplicações de matrizes unitárias e executada de forma paralela.

---

## 📌 Destaques

- Cada **thread** escolhe um valor de `a` aleatório (coprimo de `N`)
- Cada `a` é testado por múltiplas tentativas (`TENTATIVAS_POR_A`)
- A **parte quântica simulada** (produto de matrizes) é temporizada separadamente
- A lógica de estimativa de `r` e tentativa de fatoração completa o algoritmo
- O programa **para automaticamente** ao encontrar fatores não triviais
- Resultados são salvos no CSV: `shor_resultados.csv`

---

## 🛠️ Compilação

Requisitos:

- Compilador C++ com suporte a **OpenMP** (ex: MinGW-w64, GCC)

### Comando:

```bash
g++ -fopenmp -O2 main.cpp shor_quantum.cpp -o shor_parallel.exe -std=c++17
```

---

## ▶️ Execução

```bash
./shor_parallel.exe <N>
```

**Exemplo:**

```bash
./shor_parallel.exe 35
```

---

## 🔧 Parâmetros fixos no código

| Parâmetro               | Valor Padrão | Descrição |
|------------------------|--------------|-----------|
| `NUM_A`                | 8            | Número de threads (valores de `a` testados em paralelo) |
| `TENTATIVAS_POR_A`     | 5            | Quantas vezes tentar cada `a` |
| `num_threads_inner`    | 4            | Threads internas para multiplicação de matrizes |

Você pode ajustar esses valores no `main.cpp`.

---

## 📊 Saída CSV

Arquivo: `shor_resultados.csv`

| thread | a | tentativa | medicao | binario | fase | r | x | f1 | f2 | tempo | success |
|--------|---|-----------|---------|---------|------|---|---|----|----|-------|---------|
| 0      | 2 | 1         | 5       | 00000101| 0.3125 | 8 | 4 | 3 | 5 | 0.00123 | 1 |
```