# 🧮 Shor Modular – Simulação Clássica com OpenMP

Este projeto implementa a **simulação clássica da parte quântica do algoritmo de Shor**, paralelizada com **OpenMP**, utilizando C++. Ele executa todo o ciclo do algoritmo, incluindo:

- Parte quântica (simulada) com medição
- Estimativa do período `r`
- Tentativa de fatoração de `N`
- Registro dos resultados em um arquivo `.csv`

---

## ⚙️ Requisitos

- Compilador C++ com suporte a OpenMP (ex: MinGW-w64, GCC, Clang)
- Terminal para compilar e rodar

---

## 📁 Estrutura dos Arquivos

- `main.cpp` — Executa o algoritmo completo de Shor (paralelismo, tentativas, salvamento)
- `shor_quantum.cpp` — Parte quântica do algoritmo, modular e paralelizada com OpenMP
- `shor_quantum.h` — Header da simulação quântica

---

## 🛠️ Compilação

Para compilar com OpenMP:

```bash
g++ -fopenmp -O2 main.cpp shor_quantum.cpp -o shor_modular.exe
```

---

## ▶️ Execução

```bash
./shor_modular.exe <N>
```

**Exemplo:**
```bash
./shor_modular.exe 35
```

---

## 🔄 Parâmetros e Comportamento

- `N` é o número a ser fatorado (passado via terminal)
- `a` é gerado aleatoriamente e **trocado a cada 5 tentativas**
- Total de tentativas: `MAX_TRIES = 100`
- Se um par de fatores válidos for encontrado, o algoritmo para

---

## 📊 Resultado

Um arquivo chamado `shor_resultados.csv` será gerado, contendo:

| tentativa | a  | medicao | binario   | fase   | r  | tempo (s) |
|-----------|----|---------|-----------|--------|----|-----------|
| 1         | 2  | 5       | 00000101  | 0.3125 | 13 | 0.00213   |
| ...       |    |         |           |        |    |           |

---

## 📌 Observações

- A **única parte temporizada** é a simulação da parte quântica (`U @ state` com OpenMP).
- Isso permite comparação direta com o tempo de execução do circuito real no **IBM Quantum / Qiskit** (sem incluir fila ou tempo de transpilação).
