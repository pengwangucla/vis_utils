import os

# Detect whether we are using MPI by looking at the environment variables.
# With some abuse, we also assign USING_MPI=False if single GPU is used.
# This is to avoid distributed optimizer for single GPU training, which may
# cause some problems when sampled softmax loss is employed.
if "PMI_SIZE" in os.environ:
    MPI_SIZE = int(os.environ["PMI_SIZE"])
    MPI_RANK = int(os.environ["PMI_RANK"])
    if MPI_SIZE > 1:
        USING_MPI = True
    else:
        USING_MPI = False
else:
    MPI_SIZE = 1
    MPI_RANK = 0
    USING_MPI = False

# List of GPUs available to this process. When using MPI, only one GPU is used
# regardless of number of visible GPUs.
if os.environ.get("CUDA_VISIBLE_DEVICES", "NoDevFiles") == "NoDevFiles":
    NUM_GPUS = 0
else:
    NUM_GPUS = len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))

GPUS = ["/gpu:{}".format(gpu) for gpu in range(NUM_GPUS)]

MPI_RANK_0 = not USING_MPI or (USING_MPI and MPI_RANK == 0)

UNK_SYMBOL = "<OOV>"
NO_SYMBOL = "<NONE>"
START_SYMBOL = "<S>"
END_SYMBOL = "</S>"
