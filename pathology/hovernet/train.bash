#!/usr/bin/env bash -l

# Read command line arguments.
# Initialize our own variables:
OUT_DIR="${HOME}/out/hovernet"
DATA_DIR="${HOME}/data/CoNSeP"
verbose=0

show_help () {
    echo -e " train hovernet from scratch.\n"
    echo -e "\nUsage: $0 [-d <data_dir>] [-f <out_dir>] [-v] [-h] \n" 
    echo "options:"
    echo -e "-d <data_dir>\tCoNSeP Data set directory. Default: ${DATA_DIR}"
    echo -e "-o <out_dir>\tHovernet training output directory. Default: ${OUT_DIR}"
    echo -e "-h\tPrint this Help."
    echo -e "-v\tVerbose mode."
    echo -e ""
}

# Reset in case getopts has been used previously in the shell.
OPTIND=1
while getopts "h?vd:o:" opt; do
    case "$opt" in
	h|\?)
	    show_help
	    exit 0
	    ;;
	v)  verbose=1
	    ;;
	d)  DATA_DIR=${OPTARG}
	    ;;
	o)  OUT_DIR=${OPTARG}
	    ;;
    esac
done
shift $((OPTIND-1))
[ "${1:-}" = "--" ] && shift

# Set up environment.
LOGS_DIR="${OUT_DIR}/logs"
conda activate monaipath
cuda_available=$(python -c "import torch; print(torch.cuda.is_available())")
echo "DATA_DIR = ${DATA_DIR}"
echo "OUT_DIR = ${OUT_DIR}"
echo "Logs are written to ${LOGS_DIR}"

DATA_DIR_OPTS="--root ${DATA_DIR}/Prepared"
TRAINING_CMD="training.py ${DATA_DIR_OPTS}"

# Train.
if [[ ${cuda_available} == "True" ]]; then
    # Train a HoVerNet model on multi-GPU with default arguments
    num_gpus=$(python -c "import torch; print(torch.cuda.device_count())")  
    echo "${num_gpus}-GPU training"
    echo "Stage 0"
    torchrun --nnodes=${num_gpus} --nproc_per_node=2 ${TRAINING_CMD}
    echo "Stage 1"
    torchrun --nnodes=${num_gpus} --nproc_per_node=2 ${TRAINING_CMD} --stage 1
else
    echo "CPU training"
    # Train a HoVerNet model on single-GPU or CPU-only (replace with your own ckpt path)
    export CUDA_VISIBLE_DEVICES=0
    echo "Stage 0"
    python ${TRAINING_CMD} --stage 0 --ep 50 --bs 16 --log-dir ${LOGS_DIR}
    echo "Stage 1"
    python ${TRAINING_CMD} --stage 1 --ep 50 --bs 16 --log-dir ${LOGS_DIR} \
	   --ckpt ${LOGS_DIR}/stage0/model.pt
fi
