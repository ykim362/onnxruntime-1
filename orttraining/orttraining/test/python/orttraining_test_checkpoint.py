import subprocess
import os
import shutil
import sys
import torch

def makedir(checkpoint_dir):
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir, exist_ok = True)

if __name__ == '__main__':
    checkpoint_dir = 'checkpoint_dir/'
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir, exist_ok = True)

    ngpus = torch.cuda.device_count()

    # save all checkpoint files (pre-checkpoint)
    assert subprocess.call([sys.executable, 'orttraining_test_save_checkpoint.py', 'single_node_full_precision']) == 0
    assert subprocess.call([sys.executable, 'orttraining_test_save_checkpoint.py', 'single_node_mixed_precision']) == 0
    assert subprocess.call(['mpirun', '-n', str(ngpus), '-x', 'NCCL_DEBUG=INFO', sys.executable, 'orttraining_test_save_checkpoint.py', 'data_parallelism_full_precision']) == 0
    assert subprocess.call(['mpirun', '-n', str(ngpus), '-x', 'NCCL_DEBUG=INFO', sys.executable, 'orttraining_test_save_checkpoint.py', 'data_parallelism_mixed_precision']) == 0
    assert subprocess.call(['mpirun', '-n', str(ngpus), '-x', 'NCCL_DEBUG=INFO', sys.executable, 'orttraining_test_save_checkpoint.py', 'distributed_zero_full_precision']) == 0
    assert subprocess.call(['mpirun', '-n', str(ngpus), '-x', 'NCCL_DEBUG=INFO', sys.executable, 'orttraining_test_save_checkpoint.py', 'distributed_zero_mixed_precision']) == 0

    # load checkpoint files (post-checkpoint)
    # going to single node trainer
    assert subprocess.call([sys.executable, 'orttraining_test_load_checkpoint.py', 'test_load_from_single_node_full_precision_into_single_node_full_precision']) == 0
    assert subprocess.call([sys.executable, 'orttraining_test_load_checkpoint.py', 'test_load_from_single_node_mixed_precision_into_single_node_mixed_precision']) == 0
    assert subprocess.call([sys.executable, 'orttraining_test_load_checkpoint.py', 'test_load_from_single_node_mixed_precision_into_single_node_full_precision']) == 0
    assert subprocess.call([sys.executable, 'orttraining_test_load_checkpoint.py', 'test_load_from_single_node_full_precision_into_single_node_mixed_precision']) == 0
    assert subprocess.call([sys.executable, 'orttraining_test_load_checkpoint.py', 'test_load_from_data_parallelism_full_precision_into_single_node_full_precision']) == 0
    assert subprocess.call([sys.executable, 'orttraining_test_load_checkpoint.py', 'test_load_from_data_parallelism_mixed_precision_into_single_node_full_precision']) == 0
    assert subprocess.call([sys.executable, 'orttraining_test_load_checkpoint.py', 'test_load_from_data_parallelism_mixed_precision_into_single_node_mixed_precision']) == 0
    assert subprocess.call([sys.executable, 'orttraining_test_load_checkpoint.py', 'test_load_from_data_parallelism_full_precision_into_single_node_mixed_precision']) == 0
    assert subprocess.call([sys.executable, 'orttraining_test_load_checkpoint.py', 'test_load_from_distributed_zero_full_precision_into_single_node_full_precision']) == 0
    assert subprocess.call([sys.executable, 'orttraining_test_load_checkpoint.py', 'test_load_from_distributed_zero_mixed_precision_into_single_node_full_precision']) == 0
    assert subprocess.call([sys.executable, 'orttraining_test_load_checkpoint.py', 'test_load_from_distributed_zero_mixed_precision_into_single_node_mixed_precision']) == 0
    assert subprocess.call([sys.executable, 'orttraining_test_load_checkpoint.py', 'test_load_from_distributed_zero_full_precision_into_single_node_mixed_precision']) == 0

    # going to data parallel trainer
    assert subprocess.call(['mpirun', '-n', str(ngpus), '-x', 'NCCL_DEBUG=INFO', sys.executable, 'orttraining_test_load_checkpoint.py', 'test_load_from_single_node_full_precision_into_data_parallelism_full_precision']) == 0
    assert subprocess.call(['mpirun', '-n', str(ngpus), '-x', 'NCCL_DEBUG=INFO', sys.executable, 'orttraining_test_load_checkpoint.py', 'test_load_from_single_node_mixed_precision_into_data_parallelism_full_precision']) == 0
    assert subprocess.call(['mpirun', '-n', str(ngpus), '-x', 'NCCL_DEBUG=INFO', sys.executable, 'orttraining_test_load_checkpoint.py', 'test_load_from_single_node_mixed_precision_into_data_parallelism_mixed_precision']) == 0
    assert subprocess.call(['mpirun', '-n', str(ngpus), '-x', 'NCCL_DEBUG=INFO', sys.executable, 'orttraining_test_load_checkpoint.py', 'test_load_from_single_node_full_precision_into_data_parallelism_mixed_precision']) == 0
    assert subprocess.call(['mpirun', '-n', str(ngpus), '-x', 'NCCL_DEBUG=INFO', sys.executable, 'orttraining_test_load_checkpoint.py', 'test_load_from_data_parallelism_full_precision_into_data_parallelism_full_precision']) == 0
    assert subprocess.call(['mpirun', '-n', str(ngpus), '-x', 'NCCL_DEBUG=INFO', sys.executable, 'orttraining_test_load_checkpoint.py', 'test_load_from_data_parallelism_mixed_precision_into_data_parallelism_full_precision']) == 0
    assert subprocess.call(['mpirun', '-n', str(ngpus), '-x', 'NCCL_DEBUG=INFO', sys.executable, 'orttraining_test_load_checkpoint.py', 'test_load_from_data_parallelism_mixed_precision_into_data_parallelism_mixed_precision']) == 0
    assert subprocess.call(['mpirun', '-n', str(ngpus), '-x', 'NCCL_DEBUG=INFO', sys.executable, 'orttraining_test_load_checkpoint.py', 'test_load_from_data_parallelism_full_precision_into_data_parallelism_mixed_precision']) == 0
    assert subprocess.call(['mpirun', '-n', str(ngpus), '-x', 'NCCL_DEBUG=INFO', sys.executable, 'orttraining_test_load_checkpoint.py', 'test_load_from_distributed_zero_full_precision_into_data_parallelism_full_precision']) == 0
    assert subprocess.call(['mpirun', '-n', str(ngpus), '-x', 'NCCL_DEBUG=INFO', sys.executable, 'orttraining_test_load_checkpoint.py', 'test_load_from_distributed_zero_mixed_precision_into_data_parallelism_full_precision']) == 0
    assert subprocess.call(['mpirun', '-n', str(ngpus), '-x', 'NCCL_DEBUG=INFO', sys.executable, 'orttraining_test_load_checkpoint.py', 'test_load_from_distributed_zero_mixed_precision_into_data_parallelism_mixed_precision']) == 0
    assert subprocess.call(['mpirun', '-n', str(ngpus), '-x', 'NCCL_DEBUG=INFO', sys.executable, 'orttraining_test_load_checkpoint.py', 'test_load_from_distributed_zero_full_precision_into_data_parallelism_mixed_precision']) == 0

    # going to distributed zero trainer
    assert subprocess.call(['mpirun', '-n', str(ngpus), '-x', 'NCCL_DEBUG=INFO', sys.executable, 'orttraining_test_load_checkpoint.py', 'test_load_from_single_node_full_precision_into_distributed_zero_full_precision']) == 0
    assert subprocess.call(['mpirun', '-n', str(ngpus), '-x', 'NCCL_DEBUG=INFO', sys.executable, 'orttraining_test_load_checkpoint.py', 'test_load_from_single_node_mixed_precision_into_distributed_zero_full_precision']) == 0
    assert subprocess.call(['mpirun', '-n', str(ngpus), '-x', 'NCCL_DEBUG=INFO', sys.executable, 'orttraining_test_load_checkpoint.py', 'test_load_from_single_node_mixed_precision_into_distributed_zero_mixed_precision']) == 0
    assert subprocess.call(['mpirun', '-n', str(ngpus), '-x', 'NCCL_DEBUG=INFO', sys.executable, 'orttraining_test_load_checkpoint.py', 'test_load_from_single_node_full_precision_into_distributed_zero_mixed_precision']) == 0
    assert subprocess.call(['mpirun', '-n', str(ngpus), '-x', 'NCCL_DEBUG=INFO', sys.executable, 'orttraining_test_load_checkpoint.py', 'test_load_from_data_parallelism_full_precision_into_distributed_zero_full_precision']) == 0
    assert subprocess.call(['mpirun', '-n', str(ngpus), '-x', 'NCCL_DEBUG=INFO', sys.executable, 'orttraining_test_load_checkpoint.py', 'test_load_from_data_parallelism_mixed_precision_into_distributed_zero_full_precision']) == 0
    assert subprocess.call(['mpirun', '-n', str(ngpus), '-x', 'NCCL_DEBUG=INFO', sys.executable, 'orttraining_test_load_checkpoint.py', 'test_load_from_data_parallelism_mixed_precision_into_distributed_zero_mixed_precision']) == 0
    assert subprocess.call(['mpirun', '-n', str(ngpus), '-x', 'NCCL_DEBUG=INFO', sys.executable, 'orttraining_test_load_checkpoint.py', 'test_load_from_data_parallelism_full_precision_into_distributed_zero_mixed_precision']) == 0
    assert subprocess.call(['mpirun', '-n', str(ngpus), '-x', 'NCCL_DEBUG=INFO', sys.executable, 'orttraining_test_load_checkpoint.py', 'test_load_from_distributed_zero_full_precision_into_distributed_zero_full_precision']) == 0
    assert subprocess.call(['mpirun', '-n', str(ngpus), '-x', 'NCCL_DEBUG=INFO', sys.executable, 'orttraining_test_load_checkpoint.py', 'test_load_from_distributed_zero_mixed_precision_into_distributed_zero_full_precision']) == 0
    assert subprocess.call(['mpirun', '-n', str(ngpus), '-x', 'NCCL_DEBUG=INFO', sys.executable, 'orttraining_test_load_checkpoint.py', 'test_load_from_distributed_zero_mixed_precision_into_distributed_zero_mixed_precision']) == 0
    assert subprocess.call(['mpirun', '-n', str(ngpus), '-x', 'NCCL_DEBUG=INFO', sys.executable, 'orttraining_test_load_checkpoint.py', 'test_load_from_distributed_zero_full_precision_into_distributed_zero_mixed_precision']) == 0

    shutil.rmtree(checkpoint_dir)