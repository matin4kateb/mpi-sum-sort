# MPI Parallel Sum and Sort

**Author:** Mohammad Matin Kateb  
**Course:** Distributed Systems (Shahrekord University)  

---

## Overview

This project implements a **distributed program using MPI** to perform both **parallel summation** and **parallel sorting** on very large datasets.  
It is designed as a coursework assignment for the **Distributed Systems** course and demonstrates practical knowledge of **MPI-based parallel computation**, **data distribution**, and **synchronization across multiple processes**.

The program can efficiently handle datasets of **100 million elements or more** and is suitable for execution across **3 or more independent systems or virtual machines**.

---

## Features

1. **Parallel Summation**
   - The large array is divided among all MPI processes.
   - Each process computes the sum of its local portion.
   - Local sums are gathered at the root process to compute the **total sum**.

2. **Parallel Sorting (Sample Sort)**
   - Each process first sorts its local data.
   - Representative samples are gathered at the root to determine **global pivots**.
   - Data is partitioned according to pivots and redistributed using **All-to-All communication**.
   - Final local sorting produces a **globally sorted array**.

3. **Scalability**
   - Load-balanced data distribution handles uneven chunk sizes.
   - Communication minimized to pivot exchange and partition redistribution.
   - Works efficiently on distributed multi-node environments.

---

## Requirements

- Python 3.x  
- [mpi4py](https://mpi4py.readthedocs.io/en/stable/) library  
- MPI implementation (e.g., OpenMPI, MPICH)  
- Sufficient RAM to handle large arrays per node  

---

## How to Run

1. Install `mpi4py` if not already installed:
```bash
pip install mpi4py
````

2. Run the program on multiple processes (example with 3 nodes/processes):

```bash
mpirun -np 3 python mpi_sum_sort.py
```

> Note: For very large datasets (100M elements), ensure each node/VM has enough memory.

---

## Output

* Total sum of the array (printed by the root process)
* Optional verification of the sorted array:

  * First 10 elements
  * Last 10 elements

Example:

```
Total sum of array: 49999987.123456
Sorting complete. Verification:
First 10 elements: [0.0000123 0.0000456 ...]
Last 10 elements: [0.9999543 0.9999876 ...]
```

---

## Project Structure

* `mpi_sum_sort.py` — Main program implementing distributed sum and parallel sort.
* `README.md` — Project description, requirements, and usage instructions.

---

## Notes

* This implementation uses **Sample Sort** for distributed sorting.
* Data gathering at the root is **optional**; for extremely large arrays, fully distributed processing is recommended to avoid memory bottlenecks.
* Designed to demonstrate **core MPI principles**, **process communication**, and **distributed computation workflows** for academic purposes.

---
