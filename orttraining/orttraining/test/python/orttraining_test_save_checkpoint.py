import os
import pickle
import argparse
from itertools import islice

import torch
import torch.distributed as dist

from onnxruntime import set_seed
from onnxruntime.training import amp, checkpoint, optim, orttrainer
from orttraining_test_orttrainer_frontend import _load_pytorch_transformer_model
from onnxruntime.capi._pybind_state import set_cuda_device_id, get_mpi_context_world_rank, get_mpi_context_world_size

def train(trainer, train_data, batcher_fn, total_batch_steps = 5, seed = 1):
    for i in range(total_batch_steps):
        torch.manual_seed(seed)
        set_seed(seed)
        data, targets = batcher_fn(train_data, i*35)
        trainer.train_step(data, targets)

def makedir(checkpoint_dir):
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir, exist_ok = True)

def save(trainer, checkpoint_dir, state_dict_key_name = 'state_dict'):
    # save current model parameters as a checkpoint
    makedir(checkpoint_dir)
    checkpoint.experimental_save_checkpoint(trainer, checkpoint_dir)
    state_dict = checkpoint.experimental_state_dict(trainer)
    pickle.dump({state_dict_key_name : state_dict}, open(checkpoint_dir+state_dict_key_name+'.pkl', "wb"))

def chunkify(sequence, num_chunks):
    quo, rem = divmod(len(sequence), num_chunks)
    return (sequence[i * quo + min(i, rem):(i + 1) * quo + min(i + 1, rem)] for i in range(num_chunks))

def distributed_setup(save_function):
    def setup():
        world_rank = get_mpi_context_world_rank()
        world_size = get_mpi_context_world_size()
        device = 'cuda:' + str(world_rank)

        os.environ['RANK'] = str(world_rank)
        os.environ['WORLD_SIZE'] = str(world_size)
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29500'

        set_cuda_device_id(world_rank)

        dist.init_process_group(backend='nccl', world_size=world_size, rank=world_rank)
        save_function(world_rank, world_size, device)
    return setup


def single_node_full_precision(device = 'cuda', checkpoint_dir = 'checkpoint_dir/single_node/full_precision/'):
    learning_rate = 0.1
    seed = 1

    torch.manual_seed(seed)
    set_seed(seed)

    # PyTorch transformer model as example
    opts = {'device' : {'id' : device},
            'debug' : {'deterministic_compute': True}}
    opts = orttrainer.ORTTrainerOptions(opts)
    optim_config = optim.LambConfig(lr=learning_rate)
    model, model_desc, loss_fn, batcher_fn, train_data, _, _ = _load_pytorch_transformer_model(device)
    trainer = orttrainer.ORTTrainer(model, model_desc, optim_config, loss_fn=loss_fn, options=opts)

    # run train steps
    train(trainer, train_data, batcher_fn)

    # save current model parameters as a checkpoint
    save(trainer, checkpoint_dir)

def single_node_mixed_precision(device = 'cuda', checkpoint_dir = 'checkpoint_dir/single_node/mixed_precision/'):
    learning_rate = 0.1
    seed = 1

    torch.manual_seed(seed)
    set_seed(seed)

    # PyTorch transformer model as example
    opts = {
                'device' : {'id' : device},
                'mixed_precision':
                {
                    'enabled': True,
                    'loss_scaler': amp.DynamicLossScaler()
                },
                'debug' : {'deterministic_compute': True}
            }
    opts = orttrainer.ORTTrainerOptions(opts)
    optim_config = optim.LambConfig(lr=learning_rate)
    model, model_desc, loss_fn, batcher_fn, train_data, _, _ = _load_pytorch_transformer_model(device)
    trainer = orttrainer.ORTTrainer(model, model_desc, optim_config, loss_fn=loss_fn, options=opts)

    # run train steps
    train(trainer, train_data, batcher_fn)

    # save current model parameters as a checkpoint
    save(trainer, checkpoint_dir)

@distributed_setup
def data_parallelism_full_precision(world_rank, world_size, device, checkpoint_dir = 'checkpoint_dir/data_parallelism/full_precision/'):
    learning_rate = 0.1
    seed = 1

    torch.manual_seed(seed)
    set_seed(seed)

    # PyTorch transformer model as example
    opts = {
                'device' : {'id' : device},
                'distributed' :
                {
                    'world_rank' : world_rank,
                    'world_size' : world_size,
                    'allreduce_post_accumulation' : True
                },
                'debug' : {'deterministic_compute': True}
            }
    opts = orttrainer.ORTTrainerOptions(opts)
    optim_config = optim.LambConfig(lr=learning_rate)
    model, model_desc, loss_fn, batcher_fn, train_data, _, _ = _load_pytorch_transformer_model(device)
    train_chunk = next(islice(chunkify(train_data, world_size), world_rank, None))
    trainer = orttrainer.ORTTrainer(model, model_desc, optim_config, loss_fn=loss_fn, options=opts)

    # run train steps
    train(trainer, train_chunk, batcher_fn)

    # save current model parameters as a checkpoint
    if world_rank == 0:
        save(trainer, checkpoint_dir)

@distributed_setup
def data_parallelism_mixed_precision(world_rank, world_size, device, checkpoint_dir = 'checkpoint_dir/data_parallelism/mixed_precision/'):
    learning_rate = 0.1
    seed = 1

    torch.manual_seed(seed)
    set_seed(seed)

    # PyTorch transformer model as example
    opts = {
                'device' : {'id' : device},
                'mixed_precision':
                {
                    'enabled': True,
                    'loss_scaler': amp.DynamicLossScaler()
                },
                'distributed' :
                {
                    'world_rank' : world_rank,
                    'world_size' : world_size,
                    'allreduce_post_accumulation' : True
                },
                'debug' : {'deterministic_compute': True}
            }
    opts = orttrainer.ORTTrainerOptions(opts)
    optim_config = optim.LambConfig(lr=learning_rate)
    model, model_desc, loss_fn, batcher_fn, train_data, _, _ = _load_pytorch_transformer_model(device)
    train_chunk = next(islice(chunkify(train_data, world_size), world_rank, None))
    trainer = orttrainer.ORTTrainer(model, model_desc, optim_config, loss_fn=loss_fn, options=opts)

    # run train steps
    train(trainer, train_chunk, batcher_fn)

    # save current model parameters as a checkpoint
    if world_rank == 0:
        save(trainer, checkpoint_dir)

@distributed_setup
def distributed_zero_full_precision(world_rank, world_size, device, checkpoint_dir = 'checkpoint_dir/distributed_zero/full_precision/'):
    learning_rate = 0.1
    seed = 1

    torch.manual_seed(seed)
    set_seed(seed)

    # PyTorch transformer model as example
    opts = {
                'device' : {'id' : device},
                'distributed' :
                {
                    'world_rank' : world_rank,
                    'world_size' : world_size,
                    'allreduce_post_accumulation' : True,
                    'deepspeed_zero_optimization':
                    {
                        'stage': 1
                    }
                },
                'debug' : {'deterministic_compute': True}
            }
    opts = orttrainer.ORTTrainerOptions(opts)
    optim_config = optim.LambConfig(lr=learning_rate)
    model, model_desc, loss_fn, batcher_fn, train_data, _, _ = _load_pytorch_transformer_model(device)
    train_chunk = next(islice(chunkify(train_data, world_size), world_rank, None))
    trainer = orttrainer.ORTTrainer(model, model_desc, optim_config, loss_fn=loss_fn, options=opts)

    # run train steps
    train(trainer, train_chunk, batcher_fn)

    # save current model parameters as a checkpoint
    save(trainer, checkpoint_dir, 'state_dict_'+str(world_rank))

@distributed_setup
def distributed_zero_mixed_precision(world_rank, world_size, device, checkpoint_dir = 'checkpoint_dir/distributed_zero/mixed_precision/'):
    learning_rate = 0.1
    seed = 1

    torch.manual_seed(seed)
    set_seed(seed)

    # PyTorch transformer model as example
    opts = {
                'device' : {'id' : device},
                'mixed_precision':
                {
                    'enabled': True,
                    'loss_scaler': amp.DynamicLossScaler()
                },
                'distributed' :
                {
                    'world_rank' : world_rank,
                    'world_size' : world_size,
                    'allreduce_post_accumulation' : True,
                    'deepspeed_zero_optimization':
                    {
                        'stage': 1
                    }
                },
                'debug' : {'deterministic_compute': True}
            }
    opts = orttrainer.ORTTrainerOptions(opts)
    optim_config = optim.LambConfig(lr=learning_rate)
    model, model_desc, loss_fn, batcher_fn, train_data, _, _ = _load_pytorch_transformer_model(device)
    train_chunk = next(islice(chunkify(train_data, world_size), world_rank, None))
    trainer = orttrainer.ORTTrainer(model, model_desc, optim_config, loss_fn=loss_fn, options=opts)

    # run train steps
    train(trainer, train_chunk, batcher_fn)

    # save current model parameters as a checkpoint
    save(trainer, checkpoint_dir, 'state_dict_'+str(world_rank))

if __name__ == '__main__':
    function_map = {
        'single_node_full_precision': single_node_full_precision,
        'single_node_mixed_precision': single_node_mixed_precision,
        'data_parallelism_full_precision': data_parallelism_full_precision,
        'data_parallelism_mixed_precision': data_parallelism_mixed_precision,
        'distributed_zero_full_precision': distributed_zero_full_precision,
        'distributed_zero_mixed_precision': distributed_zero_mixed_precision
    }
    parser = argparse.ArgumentParser(description='Save states of trainers')
    parser.add_argument('scenario', choices=function_map.keys(), help='training scenario to save states')
    args = parser.parse_args()
    function_map[args.scenario]()