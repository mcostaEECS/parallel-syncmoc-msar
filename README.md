# Parallel SyncMoC MS-AR Simulator

This repository contains a high-performance simulation framework for **multi-temporal change detection** using the **Markov-Switching Autoregressive (MS-AR)** model, designed with a **Synchronous Model of Computation (Sync-MoC)** and implemented in **Haskell with ForSyDe.Shallow**.

> Includes energy-aware parallelism analysis, performance metrics, and visualization scripts.

---

## 🔍 Overview

This simulator executes parallel change detection filters across multiple time-lagged SAR images, leveraging:

- Modular **MS-AR(k)** autoregressive filters
- Lazy Random Walk-based optimization
- A **parallel skeleton-based architecture**
- Real-time visualization with **ThreadScope**
- Energy profiling using `powerstat` + RTS

---

## 🗂 Directory Structure

```
parallel-syncmoc-msar/
│
├── analysis/           # Python scripts for power and efficiency plots (e.g., Fig.4b)
├── data/test6/         # Input SAR data (CARABAS II mission format)
├── main_simulator/     # Main simulation entry point (uses parMap + ForSyDe)
├── results/            # Simulation outputs and logs
├── src/                # MS-AR model (AR system, MC, anomaly detection)
├── stack.yaml          # Stack project configuration
└── README.md
```

---

## ⚙️ Requirements

- [GHC (>= 8.10)](https://www.haskell.org/ghc/)
- [Stack](https://docs.haskellstack.org)
- Linux (tested on Ubuntu 22.04)
- `powerstat` (for energy measurements)
- `threadscope` (for parallel visualization)

Install dependencies:

```bash
sudo apt install powerstat threadscope
```

---

## 🚀 Usage

### Compile

```bash
stack build
```

### Run simulation (with RTS metrics)

```bash
stack exec parallel-syncmoc-msar +RTS -N4 -s -ls
```

### Energy + Throughput analysis (optional)

```bash
python3 analysis/plot_efficiency.py results/run_power_log.csv
```

---

## 📊 Example: Performance Analysis

- **Fig. 4(a)**: ThreadScope timeline of skeleton-parallel filters
- **Fig. 4(b)**: Throughput vs. cumulative energy with convergence markers
- Optimal parallelism minimizes total energy at ~80% dataset coverage.

---

## 📘 Related Work

This project is part of an extended study from postdoctoral research at KTH EECS, focusing on:

- Real-time change detection
- Embedded energy modeling
- Parallel scheduling with formal skeletons

For further reading, refer to:  
> "Parallel Change Detection using Markov-Switching Filters on a Synchronous Model of Computation" (In prep.)

---

## 📄 License

MIT License © 2025 Marcello Costa (ITA/KTH)

---

## 🤝 Acknowledgements

Developed during the postdoctoral research at **KTH Royal Institute of Technology**, Dept. of Embedded Systems (EECS).