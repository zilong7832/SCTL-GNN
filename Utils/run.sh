#!/bin/bash --login

#SBATCH --job-name=seurat
#SBATCH --account=cmse
#SBATCH --gpus=1
#SBATCH --mem=100GB
#SBATCH --cpus-per-task=4
#SBATCH --time=3:59:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zhan2210@msu.edu
#SBATCH --array=0-39
#SBATCH --output=/mnt/home/zhan2210/result/%x-%A_%a.SLURMout

source /mnt/home/zhan2210/miniforge3/etc/profile.d/conda.sh

LOG_DIR="/mnt/home/zhan2210/output"
LOG_FILE="$LOG_DIR/seurat_job_usage.csv"
mkdir -p "$LOG_DIR"
# 若第一次写入则加表头
if [ ! -f "$LOG_FILE" ]; then
  echo "timestamp,job_id,array_task_id,dataset,repeat,fold,elapsed_hms,elapsed_min,max_rss_gb,exit_code" >> "$LOG_FILE"
fi

# Run the script
export CUDA_VISIBLE_DEVICES=0

: ${SLURM_ARRAY_TASK_ID:=0}
# 1) 定义要跑的 dataset 列表
datasets=(PBMC PBMC-Li SLN-111 SLN-208)

# 每个数据集对应 10 次 repeat (0–9)，fold 固定为 0
dataset_idx=$(( SLURM_ARRAY_TASK_ID / 10 ))   # 0,1,2
repeat=$(( SLURM_ARRAY_TASK_ID % 10 ))        # 0–9
fold=0                                         # 固定为 0

dataset=${datasets[$dataset_idx]}

# 导出环境变量供 Python 使用
export DATASET=$dataset
export REPEAT_ID=$repeat
export FOLD_ID=$fold

echo ">> Running DATASET=$dataset | REPEAT=$repeat| FOLD=$fold"

# ==== timed run (REPLACE the python line with the block below) ====
TMP_METRIC="$(mktemp)"
# /usr/bin/time -f "%e %E %M %x" -o "$TMP_METRIC" -- /mnt/home/zhan2210/miniforge3/envs/dance-env/bin/python /mnt/home/zhan2210/scProtein/code/SCTLGNN/SCTLGNN.py --calibrate

# /usr/bin/time -f "%e %E %M %x" -o "$TMP_METRIC" -- /mnt/home/zhan2210/miniforge3/envs/dance-env/bin/python /mnt/home/zhan2210/scProtein/code/scMoGNN/scMoGNN-save.py
/usr/bin/time -f "%e %E %M %x" -o "$TMP_METRIC" -- /mnt/home/zhan2210/miniforge3/envs/r-env/bin/Rscript "/mnt/home/zhan2210/scProtein/code/Seurat/Seurat v5.R"

# /usr/bin/time -f "%e %E %M %x" -o "$TMP_METRIC" -- /mnt/home/zhan2210/miniforge3/envs/scipenn_env/bin/python "/mnt/scratch/zhan2210/RNA Predict/sciPENN_codes/Experiments/myscipenn.py"
# /usr/bin/time -f "%e %E %M %x" -o "$TMP_METRIC" -- /mnt/home/zhan2210/miniforge3/envs/saverx_env/bin/python "/mnt/scratch/zhan2210/RNA Predict/ctpnet/saver-x.py" 
# /usr/bin/time -f "%e %E %M %x" -o "$TMP_METRIC" -- /mnt/home/zhan2210/miniforge3/envs/ctpnet_env/bin/python /mnt/scratch/zhan2210/RNA\ Predict/ctpnet/ctpnet.py
# /usr/bin/time -f "%e %E %M %x" -o "$TMP_METRIC" -- /mnt/home/zhan2210/miniforge3/envs/moetm/bin/python /mnt/scratch/zhan2210/RNA\ Predict/moETM/main_cross_prediction_rna_protein.py


read elapsed_sec elapsed_hms maxrss_kb exitcode < "$TMP_METRIC"
elapsed_min=$(awk -v s="$elapsed_sec" 'BEGIN{printf "%.2f", s/60}')
max_rss_gb=$(awk -v kb="$maxrss_kb" 'BEGIN{printf "%.3f", kb/1024/1024}')
rm -f "$TMP_METRIC"

# 记录到 CSV（时间戳使用 ISO 格式）
printf "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n" \
  "$(date -Is)" \
  "${SLURM_JOB_ID:-}" \
  "${SLURM_ARRAY_TASK_ID:-}" \
  "$dataset" "$repeat" "$fold" \
  "$elapsed_hms" "$elapsed_min" "$max_rss_gb" "$exitcode" >> "$LOG_FILE"
# ==== end timed run ====


# sbatch /mnt/home/zhan2210/scProtein/code/run.sh
# squeue -u $USER