#!/bin/bash
#SBATCH --job-name=bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=24
#SBATCH --mem=48gb
#SBATCH --time=24:00:00
#SBATCH --output=./reports/output_%a_%j.o 
#SBATCH --error=./reports/error_%a_%j.e


BASE_PATH=/datasets/work/ev-ca-macq/work/sho108/hydroml

cd $BASE_PATH
. .venv/bin/activate
export PYTHONPATH=$BASE_PATH
python examples/scripts/virga/validation.py $1

# example of use sbatch: sbatch examples/scripts/virga/validation.slurm toos_c_validation_silo_silo
