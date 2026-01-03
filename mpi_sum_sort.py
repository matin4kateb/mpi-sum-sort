from mpi4py import MPI
import numpy as np

def parallel_sum(data, comm):
    """
    Perform distributed summation.
    Each process computes local sum.
    Gather sums at root and compute total sum.
    """
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    local_sum = np.sum(data)
    
    local_sums = None
    if rank == 0:
        local_sums = np.empty(size, dtype='float64')
    
    comm.Gather(np.array(local_sum, dtype='float64'), local_sums, root=0)
    
    total_sum = None
    if rank == 0:
        total_sum = np.sum(local_sums)
    return total_sum

def parallel_sample_sort(local_data, comm):
    """
    Parallel sample sort:
    1. Local sort
    2. Sampling and pivot selection
    3. Partition local data
    4. All-to-all exchange
    5. Final local sort
    """
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Step 1: local sort
    local_data.sort()
    
    # Step 2: sample
    num_samples = size - 1
    sample_indices = np.linspace(0, len(local_data)-1, num_samples+2, dtype=int)[1:-1]
    local_samples = local_data[sample_indices]
    
    all_samples = None
    if rank == 0:
        all_samples = np.empty(num_samples*size, dtype=local_data.dtype)
    
    comm.Gather(local_samples, all_samples, root=0)
    
    # Step 3: pivots
    pivots = None
    if rank == 0:
        all_samples.sort()
        pivots = all_samples[::size]
    pivots = comm.bcast(pivots, root=0)
    
    # Step 4: partition
    partitions = []
    start_idx = 0
    for pivot in pivots:
        end_idx = np.searchsorted(local_data, pivot, side='right')
        partitions.append(local_data[start_idx:end_idx])
        start_idx = end_idx
    partitions.append(local_data[start_idx:])
    
    send_counts = np.array([len(part) for part in partitions], dtype=int)
    recv_counts = np.empty(size, dtype=int)
    comm.Alltoall(send_counts, recv_counts)
    
    send_data = np.concatenate(partitions)
    send_displs = np.insert(np.cumsum(send_counts), 0, 0)[:-1]
    recv_displs = np.insert(np.cumsum(recv_counts), 0, 0)[:-1]
    
    recv_data = np.empty(np.sum(recv_counts), dtype=local_data.dtype)
    
    comm.Alltoallv([send_data, send_counts, send_displs, MPI.DOUBLE],
                   [recv_data, recv_counts, recv_displs, MPI.DOUBLE])
    
    recv_data.sort()
    return recv_data

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    N = 100_000_000  # 100 million
    chunk_size = N // size
    remainder = N % size
    
    # Generate data at root
    if rank == 0:
        print("Generating large dataset...")
        data = np.random.rand(N).astype('float64')
        print("Dataset generated.")
    else:
        data = None
    
    # Scatterv setup
    local_sizes = np.array([chunk_size + 1 if i < remainder else chunk_size for i in range(size)], dtype=int)
    local_data = np.empty(local_sizes[rank], dtype='float64')
    
    if rank == 0:
        displs = np.insert(np.cumsum(local_sizes), 0, 0)[:-1]
    else:
        displs = None
    
    comm.Scatterv([data, local_sizes, displs, MPI.DOUBLE], local_data, root=0)
    
    # Parallel sum
    total_sum = parallel_sum(local_data, comm)
    if rank == 0:
        print(f"Total sum of array: {total_sum}")
    
    # Parallel sorting
    sorted_local = parallel_sample_sort(local_data, comm)
    
    # Gather sorted data at root (optional)
    sorted_sizes = np.array([len(sorted_local) for _ in range(size)], dtype=int)  # dummy for Gatherv
    recv_counts = np.array(comm.allgather(len(sorted_local)), dtype=int)  # actual sizes
    if rank == 0:
        final_sorted = np.empty(N, dtype='float64')
        displs = np.insert(np.cumsum(recv_counts), 0, 0)[:-1]
    else:
        final_sorted = None
        displs = None
    
    comm.Gatherv(sorted_local, [final_sorted, recv_counts, displs, MPI.DOUBLE], root=0)
    
    if rank == 0:
        print("Sorting complete. Verification:")
        print("First 10 elements:", final_sorted[:10])
        print("Last 10 elements:", final_sorted[-10:])

if __name__ == "__main__":
    main()
