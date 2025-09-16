import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as  np 

def format_bytes(x):
    if x < 1024:
        return f"{int(x)} B"
    elif x < 1024**2:
        return f"{x // 1024:.0f} KB"
    elif x < 1024**3:
        return f"{x // (1024**2):.0f} MB"
    else:
        return f"{x // (1024**3):.0f} GB"

# List of CSV files and labels
inter_files = [
    ("inter-eth-no-rdma.csv", "With Ethernet - Without GPUDirect RDMA (Staging through Host)"),
    ("inter-ib-rdma.csv", "With IB - With GPUDirect RDMA"),
    ("inter-ib-no-rdma.csv", "With IB - Without GPUDirect RDMA (Staging through Host)"),
]

intra_files = [
    ("intra-no-ipc.csv", "Without GPUDirect P2P (Staging through Host)"),
    #("intra-ipc.csv", "With GPUDirect P2P"),
]

host_files = [
    #("non-cuda-aware-host-eth.csv", "Not CUDA Aware MPI - With Ethernet"), Using non cuda MPI 4 
    ("host-eth.csv", "With Ethernet"),
    ("host-ib.csv", "With Infiniband")
]

# Create the plot
fig, axes = plt.subplots(nrows=1, ncols=1,figsize=(15, 6))

# Read and plot each file
for file_name, label in inter_files:
    print(file_name)
    df = pd.read_csv(file_name)
    df['Result'] = df['Result'] / 1000
    df['Value'] = df['Value'].apply(lambda x: format_bytes(x))
    axes.plot(df['Value'], df['Result'], marker='o', label=label)

def plot(ax):
    ax.legend()
    ax.grid(True)
    #ax.set_xscale('log')  # Optional: Use logarithmic x-axis if data spans large range
    ax.set_xlabel("Size in B")
    ax.set_ylabel("Bandwidth in GB/s")


# Show the plot
plot(axes)
fig.suptitle('OSU Bandwidth Benchmark - GPU2GPU - H100 SXM - 2 Nodes/1 GPU', fontsize=20)
plt.tight_layout()
fig.savefig("inter.png")

plt.close()

fig2, axes2 = plt.subplots(nrows=1, ncols=1,figsize=(15, 6))

# Read and plot each file
for file_name, label in intra_files:
    print(file_name)
    df = pd.read_csv(file_name)
    df['Result'] = df['Result'] / 1000
    df['Value'] = df['Value'].apply(lambda x: format_bytes(x))
    axes2.plot(df['Value'], df['Result'], marker='o', label=label)

plot(axes2)
fig2.suptitle('OSU Bandwidth Benchmark - GPU2GPU - H100 SXM - 2 Nodes/1 GPU', fontsize=20)
plt.tight_layout()
fig2.savefig("intra.png")

fig3, axes3 = plt.subplots(nrows=1, ncols=1,figsize=(15, 6))

# Read and plot each file
for file_name, label in host_files:
    print(file_name)
    df = pd.read_csv(file_name)
    df['Result'] = df['Result'] / 1000
    df['Value'] = df['Value'].apply(lambda x: format_bytes(x))
    axes3.plot(df['Value'], df['Result'], marker='o', label=label)

plot(axes3)
fig3.suptitle('OSU Bandwidth Benchmark - Host2Host - H100 SXM - 2 Nodes', fontsize=20)
plt.tight_layout()
fig3.savefig("host.png")

