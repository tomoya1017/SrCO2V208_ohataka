#!/bin/bash
#SBATCH --job-name=excitation
#SBATCH --output=excitation_%j.out     # 標準出力ログ
#SBATCH --error=excitation_%j.err      # エラーログ
#SBATCH --partition=F1cpu
#SBATCH -N 1
#SBATCH -n 10 # 128コアを使用


source /home/issp/materiapps/intel/python3/python3vars-3.8.6-1.sh

# 作業ディレクトリへ移動
cd $SLURM_SUBMIT_DIR

# OpenMPのスレッド数を制限
export OMP_NUM_THREADS=1
for numk in $(seq 0 2 4); do
    srun python3 excitation_afm_mpi4.py --numk "$numk"
done


# srun python3 excitation_afm_mpi4.py
