## SLURM (all GPUs for 1 yaml)

`python main_sbatch.py --time 12:00:00 --hp YAML --name NAME`

`python main_sbatch.py --time 12:00:00 --hp YAML --name 'NAME'` if NAME/YAML contains '('

`python main_sbatch.py --time 8:00:00 --nodes 4 --hp YAML --name NAME`

## SLURM (1 GPU per yaml)

### run ALL yamls in the folder until the folder is empty (also yamls that are added after startup)

`python main_sbatch.py --script run_folder --devices 0 --time 5:00:00 --folder FOLDER`

### run one yaml on each gpu

`python main_sbatch.py --script run_folder --devices 0 --time 5:00:00 --folder FOLDER --single`

`python main_sbatch.py --script run_folder --devices 0 --time 5:00:00 --folder FOLDER --preload <PRELOAD_YAML>`
where `<PRELOAD_YAML>` is the name of a yaml in the `yamls_preload` folder

`python main_sbatch.py --script run_folder --devices 0 --time 5:00:00 --folder FOLDER --preload in1k_train_test.yaml`

## SLURM probe with a full node split into single-GPU runs

`python main_sbatch.py --script run_folder --devices 0 --time 5:00:00 --folder yamls_probe`

#### preload datasets to avoid each run copying the dataset seperately (mostly required for large datasets like ImageNet1K)

`python main_sbatch.py --script run_folder --devices 0 --time 5:00:00 --folder yamls_probe --preload in1k_train_test.yaml`

## run all yamls in a folder

`python main_run_folder.py --folder FOLDER --devices DEVICE`

## Errors

- something similar to `ModuleNotFoundError: No module named 'collections.abc'`
    - something went wrong with the python instantiation on the node...use another node or exclude the node from your
      jobs
    - submit the job twice (the first one will allocate the broken node and crash; the second one will allocate a
      healthy node)
    - you can also exclude the node in the `sbatch` command but this is not implemented yet in `main_sbatch.py`