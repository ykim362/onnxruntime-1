import os
import pickle
from numpy.testing import assert_allclose
import numpy as np
import argparse
import glob

import torch
import torch.distributed as dist

import onnxruntime
from onnxruntime import set_seed
from onnxruntime.training import amp, checkpoint, optim, orttrainer
from onnxruntime.capi._pybind_state import set_cuda_device_id, get_mpi_context_world_rank, get_mpi_context_world_size
from orttraining_test_orttrainer_frontend import _load_pytorch_transformer_model

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

def load_evaluate_extract_state_dict(device, trainer_opts, checkpoint_dir):
    seed = 1
    torch.manual_seed(seed)
    set_seed(seed)

    # PyTorch transformer model setup
    learning_rate = 0.1
    optim_config = optim.LambConfig(lr=learning_rate)
    model, model_desc, loss_fn, batcher_fn, train_data, _, _ = _load_pytorch_transformer_model(device)
    trainer = orttrainer.ORTTrainer(model, model_desc, optim_config, loss_fn=loss_fn, options=trainer_opts)

    # load checkpoint into trainer
    checkpoint.experimental_load_checkpoint(trainer, checkpoint_dir)

    # run an eval step to innitialize the graph
    torch.manual_seed(seed)
    set_seed(seed)
    data, targets = batcher_fn(train_data, 0)
    trainer.eval_step(data, targets)

    return checkpoint.experimental_state_dict(trainer), model

def split_state_dict(state_dict):
    optimizer_keys = ['Moment_1_', 'Moment_2_', 'Update_Count_', 'Step_']
    split_sd = {'optimizer': {}, 'fp32_param': {}, 'fp16_param': {}}
    for k, v in state_dict.items():
        mode = 'fp32_param'
        for optim_key in optimizer_keys:
            if k.startswith(optim_key):
                mode = 'optimizer'
                break
        if k.endswith('_fp16'):
            mode = 'fp16_param'
        split_sd[mode][k] = v
    return split_sd

def split_name(name):
        name_split = name.split('_view_')
        view_num = None
        if(len(name_split) > 1):
            view_num = int(name_split[1])
        optimizer_key = ''
        fp16_key = ''
        if name_split[0].startswith('Moment_1'):
            optimizer_key = 'Moment_1_'
        elif name_split[0].startswith('Moment_2'):
            optimizer_key = 'Moment_2_'
        elif name_split[0].startswith('Update_Count'):
            optimizer_key = 'Update_Count_'
        elif name_split[0].endswith('_fp16'):
            fp16_key = '_fp16'
        param_name = name_split[0]
        if optimizer_key != '':
            param_name = param_name.split(optimizer_key)[1]
        param_name = param_name.split('_fp16')[0]
        return param_name, optimizer_key, view_num, fp16_key

def aggregate_states(aggregated_states, state_dict):
    for key, value in state_dict.items():
        weight_name, optimizer_key, view_num, fp16_key = split_name(key)
        if view_num is not None:
            # parameter is sharded
            param_name = optimizer_key + weight_name + fp16_key

            if param_name in aggregated_states and optimizer_key not in ['Update_Count_']:
                # found a previous shard of the param, concatenate shards ordered by ranks
                aggregated_states[param_name] = torch.cat((aggregated_states[param_name], value))
            else:
                aggregated_states[param_name] = value
        else:
            aggregated_states[key] = value

def test_load_from_single_node_full_precision_into_single_node_full_precision(device = 'cuda', checkpoint_dir = 'checkpoint_dir/single_node/full_precision/'):
    opts = {'device' : {'id' : device},
            'debug' : {'deterministic_compute': True}}
    
    # extract state dictionaries to compare
    state_dict_post_checkpoint, model = load_evaluate_extract_state_dict(device, orttrainer.ORTTrainerOptions(opts), checkpoint_dir)

    state = pickle.load(open(checkpoint_dir+'state_dict.pkl', 'rb'))
    state_dict_pre_checkpoint = state['state_dict']

    # compare all states
    for key, value in state_dict_pre_checkpoint.items():
        assert_allclose(value, state_dict_post_checkpoint[key])
    
    # load state into pytorch and compare
    model.load_state_dict(state_dict_pre_checkpoint, strict=False)
    state_dict_pytorch = model.state_dict()
    for key, value in state_dict_pytorch.items():
        assert_allclose(value, state_dict_pre_checkpoint[key])

def test_load_from_single_node_mixed_precision_into_single_node_full_precision(device = 'cuda', checkpoint_dir = 'checkpoint_dir/single_node/mixed_precision/'):
    opts = {'device' : {'id' : device},
            'debug' : {'deterministic_compute': True}}
    
    # extract state dictionaries to compare
    state_dict_post_checkpoint, model = load_evaluate_extract_state_dict(device, orttrainer.ORTTrainerOptions(opts), checkpoint_dir)

    state = pickle.load(open(checkpoint_dir+'state_dict.pkl', 'rb'))
    state_dict_pre_checkpoint = state['state_dict']

    # compare all states
    for key, value in state_dict_pre_checkpoint.items():
        if key.endswith('_fp16'):
            full_precision_key = key[:-5]
            assert full_precision_key in state_dict_post_checkpoint
            assert_allclose(value, state_dict_post_checkpoint[full_precision_key], atol=1e-3)
        else:
            assert_allclose(value, state_dict_post_checkpoint[key])
    
    # load state into pytorch and compare
    model.load_state_dict(state_dict_pre_checkpoint, strict=False)
    state_dict_pytorch = model.state_dict()
    for key, value in state_dict_pytorch.items():
        assert_allclose(value, state_dict_pre_checkpoint[key])

def test_load_from_single_node_mixed_precision_into_single_node_mixed_precision(device = 'cuda', checkpoint_dir = 'checkpoint_dir/single_node/mixed_precision/'):
    opts = {
                'device' : {'id' : device},
                'mixed_precision':
                {
                    'enabled': True,
                    'loss_scaler': amp.DynamicLossScaler()
                },
                'debug' : {'deterministic_compute': True}
            }
    
    # extract state dictionaries to compare
    state_dict_post_checkpoint, model = load_evaluate_extract_state_dict(device, orttrainer.ORTTrainerOptions(opts), checkpoint_dir)

    state = pickle.load(open(checkpoint_dir+'state_dict.pkl', 'rb'))
    state_dict_pre_checkpoint = state['state_dict']

    # compare all states
    for key, value in state_dict_pre_checkpoint.items():
        assert_allclose(value, state_dict_post_checkpoint[key])

def test_load_from_single_node_full_precision_into_single_node_mixed_precision(device = 'cuda', checkpoint_dir = 'checkpoint_dir/single_node/full_precision/'):
    opts = {
                'device' : {'id' : device},
                'mixed_precision':
                {
                    'enabled': True,
                    'loss_scaler': amp.DynamicLossScaler()
                },
                'debug' : {'deterministic_compute': True}
            }
    
    # extract state dictionaries to compare
    state_dict_post_checkpoint, model = load_evaluate_extract_state_dict(device, orttrainer.ORTTrainerOptions(opts), checkpoint_dir)

    state = pickle.load(open(checkpoint_dir+'state_dict.pkl', 'rb'))
    state_dict_pre_checkpoint = state['state_dict']

    # compare all states
    for key, value in state_dict_post_checkpoint.items():
        if key.endswith('_fp16'):
            full_precision_key = key[:-5]
            assert full_precision_key in state_dict_pre_checkpoint
            assert_allclose(value, state_dict_pre_checkpoint[full_precision_key], atol=1e-3)
        else:
            assert_allclose(value, state_dict_pre_checkpoint[key])

def test_load_from_data_parallelism_full_precision_into_single_node_full_precision(device = 'cuda', checkpoint_dir = 'checkpoint_dir/data_parallelism/full_precision/'):
    opts = {'device' : {'id' : device},
            'debug' : {'deterministic_compute': True}}
    
    # extract state dictionaries to compare
    state_dict_post_checkpoint, model = load_evaluate_extract_state_dict(device, orttrainer.ORTTrainerOptions(opts), checkpoint_dir)

    state = pickle.load(open(checkpoint_dir+'state_dict.pkl', 'rb'))
    state_dict_pre_checkpoint = state['state_dict']

    # compare all states
    for key, value in state_dict_pre_checkpoint.items():
        assert_allclose(value, state_dict_post_checkpoint[key])
    
    # load state into pytorch and compare
    model.load_state_dict(state_dict_pre_checkpoint, strict=False)
    state_dict_pytorch = model.state_dict()
    for key, value in state_dict_pytorch.items():
        assert_allclose(value, state_dict_pre_checkpoint[key])

def test_load_from_data_parallelism_mixed_precision_into_single_node_full_precision(device = 'cuda', checkpoint_dir = 'checkpoint_dir/data_parallelism/mixed_precision/'):
    opts = {'device' : {'id' : device},
            'debug' : {'deterministic_compute': True}}
    
    # extract state dictionaries to compare
    state_dict_post_checkpoint, model = load_evaluate_extract_state_dict(device, orttrainer.ORTTrainerOptions(opts), checkpoint_dir)

    state = pickle.load(open(checkpoint_dir+'state_dict.pkl', 'rb'))
    state_dict_pre_checkpoint = state['state_dict']

    # compare all states
    for key, value in state_dict_pre_checkpoint.items():
        if key.endswith('_fp16'):
            full_precision_key = key[:-5]
            assert full_precision_key in state_dict_post_checkpoint
            assert_allclose(value, state_dict_post_checkpoint[full_precision_key], atol=1e-3)
        else:
            assert_allclose(value, state_dict_post_checkpoint[key])
    
    # load state into pytorch and compare
    model.load_state_dict(state_dict_pre_checkpoint, strict=False)
    state_dict_pytorch = model.state_dict()
    for key, value in state_dict_pytorch.items():
        assert_allclose(value, state_dict_pre_checkpoint[key])

def test_load_from_data_parallelism_mixed_precision_into_single_node_mixed_precision(device = 'cuda', checkpoint_dir = 'checkpoint_dir/data_parallelism/mixed_precision/'):
    opts = {
                'device' : {'id' : device},
                'mixed_precision':
                {
                    'enabled': True,
                    'loss_scaler': amp.DynamicLossScaler()
                },
                'debug' : {'deterministic_compute': True}
            }
    
    # extract state dictionaries to compare
    state_dict_post_checkpoint, model = load_evaluate_extract_state_dict(device, orttrainer.ORTTrainerOptions(opts), checkpoint_dir)

    state = pickle.load(open(checkpoint_dir+'state_dict.pkl', 'rb'))
    state_dict_pre_checkpoint = state['state_dict']

    # compare all states
    for key, value in state_dict_pre_checkpoint.items():
        assert_allclose(value, state_dict_post_checkpoint[key])

def test_load_from_data_parallelism_full_precision_into_single_node_mixed_precision(device = 'cuda', checkpoint_dir = 'checkpoint_dir/data_parallelism/full_precision/'):
    opts = {
                'device' : {'id' : device},
                'mixed_precision':
                {
                    'enabled': True,
                    'loss_scaler': amp.DynamicLossScaler()
                },
                'debug' : {'deterministic_compute': True}
            }
    
    # extract state dictionaries to compare
    state_dict_post_checkpoint, model = load_evaluate_extract_state_dict(device, orttrainer.ORTTrainerOptions(opts), checkpoint_dir)

    state = pickle.load(open(checkpoint_dir+'state_dict.pkl', 'rb'))
    state_dict_pre_checkpoint = state['state_dict']

    # compare all states
    for key, value in state_dict_post_checkpoint.items():
        if key.endswith('_fp16'):
            full_precision_key = key[:-5]
            assert full_precision_key in state_dict_pre_checkpoint
            assert_allclose(value, state_dict_pre_checkpoint[full_precision_key], atol=1e-3)
        else:
            assert_allclose(value, state_dict_pre_checkpoint[key])

def test_load_from_distributed_zero_full_precision_into_single_node_full_precision(device = 'cuda', checkpoint_dir = 'checkpoint_dir/distributed_zero/full_precision/'):
    opts = {'device' : {'id' : device},
        'debug' : {'deterministic_compute': True}}
    
    # extract state dictionaries to compare
    state_dict_post_checkpoint, model = load_evaluate_extract_state_dict(device, orttrainer.ORTTrainerOptions(opts), checkpoint_dir)
    state_dict_post_checkpoint = split_state_dict(state_dict_post_checkpoint)

    num_states = len(glob.glob1(checkpoint_dir,"state_dict*"))
    optimizer_states = dict()
    for rank in range(num_states):
        state = pickle.load(open(checkpoint_dir+'state_dict_'+str(rank)+'.pkl', 'rb'))
        state_dict_pre_checkpoint = split_state_dict(state['state_dict_'+str(rank)])

        # compare all states
        # model states
        for key, value in state_dict_pre_checkpoint['fp32_param'].items():
            assert_allclose(value, state_dict_post_checkpoint['fp32_param'][key])

        # collect optimizer states for later comparison since they are sharded
        aggregate_states(optimizer_states, state_dict_pre_checkpoint['optimizer'])

    # compare optimizer states
    for key, value in optimizer_states.items():
        assert_allclose(value.reshape(state_dict_post_checkpoint['optimizer'][key].size()), state_dict_post_checkpoint['optimizer'][key])
    
    # load state into pytorch and compare
    checkpoint_files = checkpoint._list_checkpoint_files(checkpoint_dir, "ORT_checkpoint")
    agg_checkpoint = checkpoint._CombineZeroCheckpoint(checkpoint_files)
    agg_state_dict = agg_checkpoint.aggregate_checkpoints()
    model.load_state_dict(agg_state_dict, strict=False)
    state_dict_pytorch = model.state_dict()
    for key, value in state_dict_pytorch.items():
        assert_allclose(value, agg_state_dict[key])

def test_load_from_distributed_zero_mixed_precision_into_single_node_full_precision(device = 'cuda', checkpoint_dir = 'checkpoint_dir/distributed_zero/mixed_precision/'):
    opts = {'device' : {'id' : device},
        'debug' : {'deterministic_compute': True}}
    
    # extract state dictionaries to compare
    state_dict_post_checkpoint, model = load_evaluate_extract_state_dict(device, orttrainer.ORTTrainerOptions(opts), checkpoint_dir)
    state_dict_post_checkpoint = split_state_dict(state_dict_post_checkpoint)

    num_states = len(glob.glob1(checkpoint_dir,"state_dict*"))
    optimizer_states = dict()
    for rank in range(num_states):
        state = pickle.load(open(checkpoint_dir+'state_dict_'+str(rank)+'.pkl', 'rb'))
        state_dict_pre_checkpoint = split_state_dict(state['state_dict_'+str(rank)])

        # compare all states
        # fp16 states are not sharded
        for key, value in state_dict_pre_checkpoint['fp16_param'].items():
            assert_allclose(value, state_dict_post_checkpoint['fp32_param'][key[:-5]], atol=1e-3)

        # collect optimizer states for later comparison since they are sharded
        aggregate_states(optimizer_states, state_dict_pre_checkpoint['optimizer'])

    # compare optimizer states
    for key, value in optimizer_states.items():
        assert_allclose(value.reshape(state_dict_post_checkpoint['optimizer'][key].size()), state_dict_post_checkpoint['optimizer'][key])
    
    # load state into pytorch and compare
    checkpoint_files = checkpoint._list_checkpoint_files(checkpoint_dir, "ORT_checkpoint")
    agg_checkpoint = checkpoint._CombineZeroCheckpoint(checkpoint_files)
    agg_state_dict = agg_checkpoint.aggregate_checkpoints()
    model.load_state_dict(agg_state_dict, strict=False)
    state_dict_pytorch = model.state_dict()
    for key, value in state_dict_pytorch.items():
        assert_allclose(value, agg_state_dict[key])

def test_load_from_distributed_zero_mixed_precision_into_single_node_mixed_precision(device = 'cuda', checkpoint_dir = 'checkpoint_dir/distributed_zero/mixed_precision/'):
    opts = {
                'device' : {'id' : device},
                'mixed_precision':
                {
                    'enabled': True,
                    'loss_scaler': amp.DynamicLossScaler()
                },
                'debug' : {'deterministic_compute': True}
            }
    
    # extract state dictionaries to compare
    state_dict_post_checkpoint, model = load_evaluate_extract_state_dict(device, orttrainer.ORTTrainerOptions(opts), checkpoint_dir)
    state_dict_post_checkpoint = split_state_dict(state_dict_post_checkpoint)

    num_states = len(glob.glob1(checkpoint_dir,"state_dict*"))
    optimizer_states = dict()
    for rank in range(num_states):
        state = pickle.load(open(checkpoint_dir+'state_dict_'+str(rank)+'.pkl', 'rb'))
        state_dict_pre_checkpoint = split_state_dict(state['state_dict_'+str(rank)])

        # compare all states
        # fp16 states are not sharded
        for key, value in state_dict_pre_checkpoint['fp16_param'].items():
            assert_allclose(value, state_dict_post_checkpoint['fp16_param'][key])

        # collect optimizer states for later comparison since they are sharded
        aggregate_states(optimizer_states, state_dict_pre_checkpoint['optimizer'])

    # compare optimizer states
    for key, value in optimizer_states.items():
        assert_allclose(value.reshape(state_dict_post_checkpoint['optimizer'][key].size()), state_dict_post_checkpoint['optimizer'][key])

def test_load_from_distributed_zero_full_precision_into_single_node_mixed_precision(device = 'cuda', checkpoint_dir = 'checkpoint_dir/distributed_zero/full_precision/'):
    opts = {
                'device' : {'id' : device},
                'mixed_precision':
                {
                    'enabled': True,
                    'loss_scaler': amp.DynamicLossScaler()
                },
                'debug' : {'deterministic_compute': True}
            }
    
    # extract state dictionaries to compare
    state_dict_post_checkpoint, model = load_evaluate_extract_state_dict(device, orttrainer.ORTTrainerOptions(opts), checkpoint_dir)
    state_dict_post_checkpoint = split_state_dict(state_dict_post_checkpoint)

    num_states = len(glob.glob1(checkpoint_dir,"state_dict*"))
    optimizer_states = dict()
    for rank in range(num_states):
        state = pickle.load(open(checkpoint_dir+'state_dict_'+str(rank)+'.pkl', 'rb'))
        state_dict_pre_checkpoint = split_state_dict(state['state_dict_'+str(rank)])

        # compare all states
        # fp32 states are not sharded
        for key, value in state_dict_post_checkpoint['fp16_param'].items():
            assert_allclose(value, state_dict_pre_checkpoint['fp32_param'][key[:-5]], atol=1e-3)

        # collect optimizer states for later comparison since they are sharded
        aggregate_states(optimizer_states, state_dict_pre_checkpoint['optimizer'])

    # compare optimizer states
    for key, value in optimizer_states.items():
        assert_allclose(value.reshape(state_dict_post_checkpoint['optimizer'][key].size()), state_dict_post_checkpoint['optimizer'][key])

@distributed_setup
def test_load_from_single_node_full_precision_into_data_parallelism_full_precision(world_rank, world_size, device, checkpoint_dir = 'checkpoint_dir/single_node/full_precision/'):
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
    
    # extract state dictionaries to compare
    state_dict_post_checkpoint, model = load_evaluate_extract_state_dict(device, orttrainer.ORTTrainerOptions(opts), checkpoint_dir)

    state = pickle.load(open(checkpoint_dir+'state_dict.pkl', 'rb'))
    state_dict_pre_checkpoint = state['state_dict']

    # compare all states
    for key, value in state_dict_pre_checkpoint.items():
        assert_allclose(value, state_dict_post_checkpoint[key])
    
    # load state into pytorch and compare
    model.load_state_dict(state_dict_pre_checkpoint, strict=False)
    state_dict_pytorch = model.state_dict()
    for key, value in state_dict_pytorch.items():
        assert_allclose(value, state_dict_pre_checkpoint[key])

@distributed_setup
def test_load_from_single_node_mixed_precision_into_data_parallelism_full_precision(world_rank, world_size, device, checkpoint_dir = 'checkpoint_dir/single_node/mixed_precision/'):
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
    
    # extract state dictionaries to compare
    state_dict_post_checkpoint, model = load_evaluate_extract_state_dict(device, orttrainer.ORTTrainerOptions(opts), checkpoint_dir)

    state = pickle.load(open(checkpoint_dir+'state_dict.pkl', 'rb'))
    state_dict_pre_checkpoint = state['state_dict']

    # compare all states
    for key, value in state_dict_pre_checkpoint.items():
        if key.endswith('_fp16'):
            full_precision_key = key[:-5]
            assert full_precision_key in state_dict_post_checkpoint
            assert_allclose(value, state_dict_post_checkpoint[full_precision_key], atol=1e-3)
        else:
            assert_allclose(value, state_dict_post_checkpoint[key])
    
    # load state into pytorch and compare
    model.load_state_dict(state_dict_pre_checkpoint, strict=False)
    state_dict_pytorch = model.state_dict()
    for key, value in state_dict_pytorch.items():
        assert_allclose(value, state_dict_pre_checkpoint[key])

@distributed_setup
def test_load_from_single_node_mixed_precision_into_data_parallelism_mixed_precision(world_rank, world_size, device, checkpoint_dir = 'checkpoint_dir/single_node/mixed_precision/'):
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
    
    # extract state dictionaries to compare
    state_dict_post_checkpoint, model = load_evaluate_extract_state_dict(device, orttrainer.ORTTrainerOptions(opts), checkpoint_dir)

    state = pickle.load(open(checkpoint_dir+'state_dict.pkl', 'rb'))
    state_dict_pre_checkpoint = state['state_dict']

    # compare all states
    for key, value in state_dict_pre_checkpoint.items():
        assert_allclose(value, state_dict_post_checkpoint[key])

@distributed_setup
def test_load_from_single_node_full_precision_into_data_parallelism_mixed_precision(world_rank, world_size, device, checkpoint_dir = 'checkpoint_dir/single_node/full_precision/'):
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
    
    # extract state dictionaries to compare
    state_dict_post_checkpoint, model = load_evaluate_extract_state_dict(device, orttrainer.ORTTrainerOptions(opts), checkpoint_dir)

    state = pickle.load(open(checkpoint_dir+'state_dict.pkl', 'rb'))
    state_dict_pre_checkpoint = state['state_dict']

    # compare all states
    for key, value in state_dict_post_checkpoint.items():
        if key.endswith('_fp16'):
            full_precision_key = key[:-5]
            assert full_precision_key in state_dict_pre_checkpoint
            assert_allclose(value, state_dict_pre_checkpoint[full_precision_key], atol=1e-3)
        else:
            assert_allclose(value, state_dict_pre_checkpoint[key])

@distributed_setup
def test_load_from_data_parallelism_full_precision_into_data_parallelism_full_precision(world_rank, world_size, device, checkpoint_dir = 'checkpoint_dir/data_parallelism/full_precision/'):
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
    
    # extract state dictionaries to compare
    state_dict_post_checkpoint, model = load_evaluate_extract_state_dict(device, orttrainer.ORTTrainerOptions(opts), checkpoint_dir)

    state = pickle.load(open(checkpoint_dir+'state_dict.pkl', 'rb'))
    state_dict_pre_checkpoint = state['state_dict']

    # compare all states
    for key, value in state_dict_pre_checkpoint.items():
        assert_allclose(value, state_dict_post_checkpoint[key])
    
    # load state into pytorch and compare
    model.load_state_dict(state_dict_pre_checkpoint, strict=False)
    state_dict_pytorch = model.state_dict()
    for key, value in state_dict_pytorch.items():
        assert_allclose(value, state_dict_pre_checkpoint[key])

@distributed_setup
def test_load_from_data_parallelism_mixed_precision_into_data_parallelism_full_precision(world_rank, world_size, device, checkpoint_dir = 'checkpoint_dir/data_parallelism/mixed_precision/'):
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
    
    # extract state dictionaries to compare
    state_dict_post_checkpoint, model = load_evaluate_extract_state_dict(device, orttrainer.ORTTrainerOptions(opts), checkpoint_dir)

    state = pickle.load(open(checkpoint_dir+'state_dict.pkl', 'rb'))
    state_dict_pre_checkpoint = state['state_dict']

    # compare all states
    for key, value in state_dict_pre_checkpoint.items():
        if key.endswith('_fp16'):
            full_precision_key = key[:-5]
            assert full_precision_key in state_dict_post_checkpoint
            assert_allclose(value, state_dict_post_checkpoint[full_precision_key], atol=1e-3)
        else:
            assert_allclose(value, state_dict_post_checkpoint[key])
    
    # load state into pytorch and compare
    model.load_state_dict(state_dict_pre_checkpoint, strict=False)
    state_dict_pytorch = model.state_dict()
    for key, value in state_dict_pytorch.items():
        assert_allclose(value, state_dict_pre_checkpoint[key])

@distributed_setup
def test_load_from_data_parallelism_mixed_precision_into_data_parallelism_mixed_precision(world_rank, world_size, device, checkpoint_dir = 'checkpoint_dir/data_parallelism/mixed_precision/'):
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
    
    # extract state dictionaries to compare
    state_dict_post_checkpoint, model = load_evaluate_extract_state_dict(device, orttrainer.ORTTrainerOptions(opts), checkpoint_dir)

    state = pickle.load(open(checkpoint_dir+'state_dict.pkl', 'rb'))
    state_dict_pre_checkpoint = state['state_dict']

    # compare all states
    for key, value in state_dict_pre_checkpoint.items():
        assert_allclose(value, state_dict_post_checkpoint[key])

@distributed_setup
def test_load_from_data_parallelism_full_precision_into_data_parallelism_mixed_precision(world_rank, world_size, device, checkpoint_dir = 'checkpoint_dir/data_parallelism/full_precision/'):
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
    
    # extract state dictionaries to compare
    state_dict_post_checkpoint, model = load_evaluate_extract_state_dict(device, orttrainer.ORTTrainerOptions(opts), checkpoint_dir)

    state = pickle.load(open(checkpoint_dir+'state_dict.pkl', 'rb'))
    state_dict_pre_checkpoint = state['state_dict']

    # compare all states
    for key, value in state_dict_post_checkpoint.items():
        if key.endswith('_fp16'):
            full_precision_key = key[:-5]
            assert full_precision_key in state_dict_pre_checkpoint
            assert_allclose(value, state_dict_pre_checkpoint[full_precision_key], atol=1e-3)
        else:
            assert_allclose(value, state_dict_pre_checkpoint[key])

@distributed_setup
def test_load_from_distributed_zero_full_precision_into_data_parallelism_full_precision(world_rank, world_size, device, checkpoint_dir = 'checkpoint_dir/distributed_zero/full_precision/'):
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
    
    # extract state dictionaries to compare
    state_dict_post_checkpoint, model = load_evaluate_extract_state_dict(device, orttrainer.ORTTrainerOptions(opts), checkpoint_dir)
    state_dict_post_checkpoint = split_state_dict(state_dict_post_checkpoint)

    num_states = len(glob.glob1(checkpoint_dir,"state_dict*"))
    optimizer_states = dict()
    for rank in range(num_states):
        state = pickle.load(open(checkpoint_dir+'state_dict_'+str(rank)+'.pkl', 'rb'))
        state_dict_pre_checkpoint = split_state_dict(state['state_dict_'+str(rank)])

        # compare all states
        # model states
        for key, value in state_dict_pre_checkpoint['fp32_param'].items():
            assert_allclose(value, state_dict_post_checkpoint['fp32_param'][key])

        # collect optimizer states for later comparison since they are sharded
        aggregate_states(optimizer_states, state_dict_pre_checkpoint['optimizer'])

    # compare optimizer states
    for key, value in optimizer_states.items():
        assert_allclose(value.reshape(state_dict_post_checkpoint['optimizer'][key].size()), state_dict_post_checkpoint['optimizer'][key])
    
    # load state into pytorch and compare
    checkpoint_files = checkpoint._list_checkpoint_files(checkpoint_dir, "ORT_checkpoint")
    agg_checkpoint = checkpoint._CombineZeroCheckpoint(checkpoint_files)
    agg_state_dict = agg_checkpoint.aggregate_checkpoints()
    model.load_state_dict(agg_state_dict, strict=False)
    state_dict_pytorch = model.state_dict()
    for key, value in state_dict_pytorch.items():
        assert_allclose(value, agg_state_dict[key])

@distributed_setup
def test_load_from_distributed_zero_mixed_precision_into_data_parallelism_full_precision(world_rank, world_size, device, checkpoint_dir = 'checkpoint_dir/distributed_zero/mixed_precision/'):
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
    
    # extract state dictionaries to compare
    state_dict_post_checkpoint, model = load_evaluate_extract_state_dict(device, orttrainer.ORTTrainerOptions(opts), checkpoint_dir)
    state_dict_post_checkpoint = split_state_dict(state_dict_post_checkpoint)

    num_states = len(glob.glob1(checkpoint_dir,"state_dict*"))
    optimizer_states = dict()
    for rank in range(num_states):
        state = pickle.load(open(checkpoint_dir+'state_dict_'+str(rank)+'.pkl', 'rb'))
        state_dict_pre_checkpoint = split_state_dict(state['state_dict_'+str(rank)])

        # compare all states
        # fp16 states are not sharded
        for key, value in state_dict_pre_checkpoint['fp16_param'].items():
            assert_allclose(value, state_dict_post_checkpoint['fp32_param'][key[:-5]], atol=1e-3)

        # collect optimizer states for later comparison since they are sharded
        aggregate_states(optimizer_states, state_dict_pre_checkpoint['optimizer'])

    # compare optimizer states
    for key, value in optimizer_states.items():
        assert_allclose(value.reshape(state_dict_post_checkpoint['optimizer'][key].size()), state_dict_post_checkpoint['optimizer'][key])
    
    # load state into pytorch and compare
    checkpoint_files = checkpoint._list_checkpoint_files(checkpoint_dir, "ORT_checkpoint")
    agg_checkpoint = checkpoint._CombineZeroCheckpoint(checkpoint_files)
    agg_state_dict = agg_checkpoint.aggregate_checkpoints()
    model.load_state_dict(agg_state_dict, strict=False)
    state_dict_pytorch = model.state_dict()
    for key, value in state_dict_pytorch.items():
        assert_allclose(value, agg_state_dict[key])

@distributed_setup
def test_load_from_distributed_zero_mixed_precision_into_data_parallelism_mixed_precision(world_rank, world_size, device, checkpoint_dir = 'checkpoint_dir/distributed_zero/mixed_precision/'):
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
    
    # extract state dictionaries to compare
    state_dict_post_checkpoint, model = load_evaluate_extract_state_dict(device, orttrainer.ORTTrainerOptions(opts), checkpoint_dir)
    state_dict_post_checkpoint = split_state_dict(state_dict_post_checkpoint)

    num_states = len(glob.glob1(checkpoint_dir,"state_dict*"))
    optimizer_states = dict()
    for rank in range(num_states):
        state = pickle.load(open(checkpoint_dir+'state_dict_'+str(rank)+'.pkl', 'rb'))
        state_dict_pre_checkpoint = split_state_dict(state['state_dict_'+str(rank)])

        # compare all states
        # fp16 states are not sharded
        for key, value in state_dict_pre_checkpoint['fp16_param'].items():
            assert_allclose(value, state_dict_post_checkpoint['fp16_param'][key])

        # collect optimizer states for later comparison since they are sharded
        aggregate_states(optimizer_states, state_dict_pre_checkpoint['optimizer'])

    # compare optimizer states
    for key, value in optimizer_states.items():
        assert_allclose(value.reshape(state_dict_post_checkpoint['optimizer'][key].size()), state_dict_post_checkpoint['optimizer'][key])

@distributed_setup
def test_load_from_distributed_zero_full_precision_into_data_parallelism_mixed_precision(world_rank, world_size, device, checkpoint_dir = 'checkpoint_dir/distributed_zero/full_precision/'):
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
    
    # extract state dictionaries to compare
    state_dict_post_checkpoint, model = load_evaluate_extract_state_dict(device, orttrainer.ORTTrainerOptions(opts), checkpoint_dir)
    state_dict_post_checkpoint = split_state_dict(state_dict_post_checkpoint)

    num_states = len(glob.glob1(checkpoint_dir,"state_dict*"))
    optimizer_states = dict()
    for rank in range(num_states):
        state = pickle.load(open(checkpoint_dir+'state_dict_'+str(rank)+'.pkl', 'rb'))
        state_dict_pre_checkpoint = split_state_dict(state['state_dict_'+str(rank)])

        # compare all states
        # fp32 states are not sharded
        for key, value in state_dict_post_checkpoint['fp16_param'].items():
            assert_allclose(value, state_dict_pre_checkpoint['fp32_param'][key[:-5]], atol=1e-3)

        # collect optimizer states for later comparison since they are sharded
        aggregate_states(optimizer_states, state_dict_pre_checkpoint['optimizer'])

    # compare optimizer states
    for key, value in optimizer_states.items():
        assert_allclose(value.reshape(state_dict_post_checkpoint['optimizer'][key].size()), state_dict_post_checkpoint['optimizer'][key])

@distributed_setup
def test_load_from_single_node_full_precision_into_distributed_zero_full_precision(world_rank, world_size, device, checkpoint_dir = 'checkpoint_dir/single_node/full_precision/'):
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
    
    # extract state dictionaries to compare
    state_dict_post_checkpoint, model = load_evaluate_extract_state_dict(device, orttrainer.ORTTrainerOptions(opts), checkpoint_dir)
    state_dict_post_checkpoint = split_state_dict(state_dict_post_checkpoint)

    state = pickle.load(open(checkpoint_dir+'state_dict.pkl', 'rb'))
    state_dict_pre_checkpoint = split_state_dict(state['state_dict'])

    # compare all states
    # model states
    for key, value in state_dict_pre_checkpoint['fp32_param'].items():
        assert_allclose(value, state_dict_post_checkpoint['fp32_param'][key])

    # round about way of checking optimizer states. Save state dicts into temporary folder, read them and aggregate them.
    pickle.dump(state_dict_post_checkpoint, open(checkpoint_dir+'distributed_state_'+str(world_rank)+'.pkl', "wb"))
    dist.barrier()

    if world_rank == 0:
        num_states = len(glob.glob1(checkpoint_dir,"distributed_state*"))
        optimizer_states = dict()
        for rank in range(num_states):
            rank_state_dict = pickle.load(open(checkpoint_dir+'distributed_state_'+str(rank)+'.pkl', 'rb'))

            # collect optimizer states for later comparison since they are sharded
            aggregate_states(optimizer_states, rank_state_dict['optimizer'])

        # compare optimizer states
        """
        failure_case: non sharded to sharded weights fails
        for key, value in optimizer_states.items():
            assert_allclose(value.reshape(state_dict_pre_checkpoint['optimizer'][key].size()), state_dict_pre_checkpoint['optimizer'][key])
        """
    
    dist.barrier()
    os.remove(checkpoint_dir+'distributed_state_'+str(world_rank)+'.pkl')

@distributed_setup
def test_load_from_single_node_mixed_precision_into_distributed_zero_full_precision(world_rank, world_size, device, checkpoint_dir = 'checkpoint_dir/single_node/mixed_precision/'):
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
    
    # extract state dictionaries to compare
    state_dict_post_checkpoint, model = load_evaluate_extract_state_dict(device, orttrainer.ORTTrainerOptions(opts), checkpoint_dir)
    state_dict_post_checkpoint = split_state_dict(state_dict_post_checkpoint)

    state = pickle.load(open(checkpoint_dir+'state_dict.pkl', 'rb'))
    state_dict_pre_checkpoint = split_state_dict(state['state_dict'])

    # compare all states
    # model states
    for key, value in state_dict_pre_checkpoint['fp16_param'].items():
        assert_allclose(value, state_dict_post_checkpoint['fp32_param'][key[:-5]], atol=1e-3)

    # round about way of checking optimizer states. Save state dicts into temporary folder, read them and aggregate them.
    pickle.dump(state_dict_post_checkpoint, open(checkpoint_dir+'distributed_state_'+str(world_rank)+'.pkl', "wb"))
    dist.barrier()

    if world_rank == 0:
        num_states = len(glob.glob1(checkpoint_dir,"distributed_state*"))
        optimizer_states = dict()
        for rank in range(num_states):
            rank_state_dict = pickle.load(open(checkpoint_dir+'distributed_state_'+str(rank)+'.pkl', 'rb'))

            # collect optimizer states for later comparison since they are sharded
            aggregate_states(optimizer_states, rank_state_dict['optimizer'])

        # compare optimizer states
        """
        failure_case: non sharded to sharded weights fails
        for key, value in optimizer_states.items():
            assert_allclose(value.reshape(state_dict_pre_checkpoint['optimizer'][key].size()), state_dict_pre_checkpoint['optimizer'][key])
        """
    
    dist.barrier()
    os.remove(checkpoint_dir+'distributed_state_'+str(world_rank)+'.pkl')

@distributed_setup
def test_load_from_single_node_mixed_precision_into_distributed_zero_mixed_precision(world_rank, world_size, device, checkpoint_dir = 'checkpoint_dir/single_node/mixed_precision/'):
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
    
    # extract state dictionaries to compare
    state_dict_post_checkpoint, model = load_evaluate_extract_state_dict(device, orttrainer.ORTTrainerOptions(opts), checkpoint_dir)
    state_dict_post_checkpoint = split_state_dict(state_dict_post_checkpoint)

    state = pickle.load(open(checkpoint_dir+'state_dict.pkl', 'rb'))
    state_dict_pre_checkpoint = split_state_dict(state['state_dict'])

    # compare all states
    # model states
    for key, value in state_dict_pre_checkpoint['fp16_param'].items():
        assert_allclose(value, state_dict_post_checkpoint['fp16_param'][key])

    # round about way of checking optimizer states. Save state dicts into temporary folder, read them and aggregate them.
    pickle.dump(state_dict_post_checkpoint, open(checkpoint_dir+'distributed_state_'+str(world_rank)+'.pkl', "wb"))
    dist.barrier()

    if world_rank == 0:
        num_states = len(glob.glob1(checkpoint_dir,"distributed_state*"))
        optimizer_states = dict()
        for rank in range(num_states):
            rank_state_dict = pickle.load(open(checkpoint_dir+'distributed_state_'+str(rank)+'.pkl', 'rb'))

            # collect optimizer states for later comparison since they are sharded
            aggregate_states(optimizer_states, rank_state_dict['optimizer'])

        # compare optimizer states
        """
        failure_case: non sharded to sharded weights fails
        for key, value in optimizer_states.items():
            assert_allclose(value.reshape(state_dict_pre_checkpoint['optimizer'][key].size()), state_dict_pre_checkpoint['optimizer'][key])
        """
    
    dist.barrier()
    os.remove(checkpoint_dir+'distributed_state_'+str(world_rank)+'.pkl')

@distributed_setup
def test_load_from_single_node_full_precision_into_distributed_zero_mixed_precision(world_rank, world_size, device, checkpoint_dir = 'checkpoint_dir/single_node/full_precision/'):
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
    
    # extract state dictionaries to compare
    state_dict_post_checkpoint, model = load_evaluate_extract_state_dict(device, orttrainer.ORTTrainerOptions(opts), checkpoint_dir)
    state_dict_post_checkpoint = split_state_dict(state_dict_post_checkpoint)

    state = pickle.load(open(checkpoint_dir+'state_dict.pkl', 'rb'))
    state_dict_pre_checkpoint = split_state_dict(state['state_dict'])

    # compare all states
    # model states
    for key, value in state_dict_post_checkpoint['fp16_param'].items():
        assert_allclose(value, state_dict_pre_checkpoint['fp32_param'][key[:-5]], atol=1e-3)

    # round about way of checking optimizer states. Save state dicts into temporary folder, read them and aggregate them.
    pickle.dump(state_dict_post_checkpoint, open(checkpoint_dir+'distributed_state_'+str(world_rank)+'.pkl', "wb"))
    dist.barrier()

    if world_rank == 0:
        num_states = len(glob.glob1(checkpoint_dir,"distributed_state*"))
        optimizer_states = dict()
        for rank in range(num_states):
            rank_state_dict = pickle.load(open(checkpoint_dir+'distributed_state_'+str(rank)+'.pkl', 'rb'))

            # collect optimizer states for later comparison since they are sharded
            aggregate_states(optimizer_states, rank_state_dict['optimizer'])

        # compare optimizer states
        """
        failure_case: non sharded to sharded weights fails
        for key, value in optimizer_states.items():
            assert_allclose(value.reshape(state_dict_pre_checkpoint['optimizer'][key].size()), state_dict_pre_checkpoint['optimizer'][key])
        """
    
    dist.barrier()
    os.remove(checkpoint_dir+'distributed_state_'+str(world_rank)+'.pkl')

@distributed_setup
def test_load_from_data_parallelism_full_precision_into_distributed_zero_full_precision(world_rank, world_size, device, checkpoint_dir = 'checkpoint_dir/data_parallelism/full_precision/'):
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
    
    # extract state dictionaries to compare
    state_dict_post_checkpoint, model = load_evaluate_extract_state_dict(device, orttrainer.ORTTrainerOptions(opts), checkpoint_dir)
    state_dict_post_checkpoint = split_state_dict(state_dict_post_checkpoint)

    state = pickle.load(open(checkpoint_dir+'state_dict.pkl', 'rb'))
    state_dict_pre_checkpoint = split_state_dict(state['state_dict'])

    # compare all states
    # model states
    for key, value in state_dict_pre_checkpoint['fp32_param'].items():
        assert_allclose(value, state_dict_post_checkpoint['fp32_param'][key])

    # round about way of checking optimizer states. Save state dicts into temporary folder, read them and aggregate them.
    pickle.dump(state_dict_post_checkpoint, open(checkpoint_dir+'distributed_state_'+str(world_rank)+'.pkl', "wb"))
    dist.barrier()

    if world_rank == 0:
        num_states = len(glob.glob1(checkpoint_dir,"distributed_state*"))
        optimizer_states = dict()
        for rank in range(num_states):
            rank_state_dict = pickle.load(open(checkpoint_dir+'distributed_state_'+str(rank)+'.pkl', 'rb'))

            # collect optimizer states for later comparison since they are sharded
            aggregate_states(optimizer_states, rank_state_dict['optimizer'])

        # compare optimizer states
        """
        failure_case: non sharded to sharded weights fails
        for key, value in optimizer_states.items():
            assert_allclose(value.reshape(state_dict_pre_checkpoint['optimizer'][key].size()), state_dict_pre_checkpoint['optimizer'][key])
        """
    
    dist.barrier()
    os.remove(checkpoint_dir+'distributed_state_'+str(world_rank)+'.pkl')

@distributed_setup
def test_load_from_data_parallelism_mixed_precision_into_distributed_zero_full_precision(world_rank, world_size, device, checkpoint_dir = 'checkpoint_dir/data_parallelism/mixed_precision/'):
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
    
    # extract state dictionaries to compare
    state_dict_post_checkpoint, model = load_evaluate_extract_state_dict(device, orttrainer.ORTTrainerOptions(opts), checkpoint_dir)
    state_dict_post_checkpoint = split_state_dict(state_dict_post_checkpoint)

    state = pickle.load(open(checkpoint_dir+'state_dict.pkl', 'rb'))
    state_dict_pre_checkpoint = split_state_dict(state['state_dict'])

    # compare all states
    # model states
    for key, value in state_dict_pre_checkpoint['fp16_param'].items():
        assert_allclose(value, state_dict_post_checkpoint['fp32_param'][key[:-5]], atol=1e-3)

    # round about way of checking optimizer states. Save state dicts into temporary folder, read them and aggregate them.
    pickle.dump(state_dict_post_checkpoint, open(checkpoint_dir+'distributed_state_'+str(world_rank)+'.pkl', "wb"))
    dist.barrier()

    if world_rank == 0:
        num_states = len(glob.glob1(checkpoint_dir,"distributed_state*"))
        optimizer_states = dict()
        for rank in range(num_states):
            rank_state_dict = pickle.load(open(checkpoint_dir+'distributed_state_'+str(rank)+'.pkl', 'rb'))

            # collect optimizer states for later comparison since they are sharded
            aggregate_states(optimizer_states, rank_state_dict['optimizer'])

        # compare optimizer states
        """
        failure_case: non sharded to sharded weights fails
        for key, value in optimizer_states.items():
            assert_allclose(value.reshape(state_dict_pre_checkpoint['optimizer'][key].size()), state_dict_pre_checkpoint['optimizer'][key])
        """
    
    dist.barrier()
    os.remove(checkpoint_dir+'distributed_state_'+str(world_rank)+'.pkl')

@distributed_setup
def test_load_from_data_parallelism_mixed_precision_into_distributed_zero_mixed_precision(world_rank, world_size, device, checkpoint_dir = 'checkpoint_dir/data_parallelism/mixed_precision/'):
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
    
    # extract state dictionaries to compare
    state_dict_post_checkpoint, model = load_evaluate_extract_state_dict(device, orttrainer.ORTTrainerOptions(opts), checkpoint_dir)
    state_dict_post_checkpoint = split_state_dict(state_dict_post_checkpoint)

    state = pickle.load(open(checkpoint_dir+'state_dict.pkl', 'rb'))
    state_dict_pre_checkpoint = split_state_dict(state['state_dict'])

    # compare all states
    # model states
    for key, value in state_dict_pre_checkpoint['fp16_param'].items():
        assert_allclose(value, state_dict_post_checkpoint['fp16_param'][key])

    # round about way of checking optimizer states. Save state dicts into temporary folder, read them and aggregate them.
    pickle.dump(state_dict_post_checkpoint, open(checkpoint_dir+'distributed_state_'+str(world_rank)+'.pkl', "wb"))
    dist.barrier()

    if world_rank == 0:
        num_states = len(glob.glob1(checkpoint_dir,"distributed_state*"))
        optimizer_states = dict()
        for rank in range(num_states):
            rank_state_dict = pickle.load(open(checkpoint_dir+'distributed_state_'+str(rank)+'.pkl', 'rb'))

            # collect optimizer states for later comparison since they are sharded
            aggregate_states(optimizer_states, rank_state_dict['optimizer'])

        # compare optimizer states
        """
        failure_case: non sharded to sharded weights fails
        for key, value in optimizer_states.items():
            assert_allclose(value.reshape(state_dict_pre_checkpoint['optimizer'][key].size()), state_dict_pre_checkpoint['optimizer'][key])
        """
    
    dist.barrier()
    os.remove(checkpoint_dir+'distributed_state_'+str(world_rank)+'.pkl')

@distributed_setup
def test_load_from_data_parallelism_full_precision_into_distributed_zero_mixed_precision(world_rank, world_size, device, checkpoint_dir = 'checkpoint_dir/data_parallelism/full_precision/'):
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
    
    # extract state dictionaries to compare
    state_dict_post_checkpoint, model = load_evaluate_extract_state_dict(device, orttrainer.ORTTrainerOptions(opts), checkpoint_dir)
    state_dict_post_checkpoint = split_state_dict(state_dict_post_checkpoint)

    state = pickle.load(open(checkpoint_dir+'state_dict.pkl', 'rb'))
    state_dict_pre_checkpoint = split_state_dict(state['state_dict'])

    # compare all states
    # model states
    for key, value in state_dict_post_checkpoint['fp16_param'].items():
        assert_allclose(value, state_dict_pre_checkpoint['fp32_param'][key[:-5]], atol=1e-3)

    # round about way of checking optimizer states. Save state dicts into temporary folder, read them and aggregate them.
    pickle.dump(state_dict_post_checkpoint, open(checkpoint_dir+'distributed_state_'+str(world_rank)+'.pkl', "wb"))
    dist.barrier()

    if world_rank == 0:
        num_states = len(glob.glob1(checkpoint_dir,"distributed_state*"))
        optimizer_states = dict()
        for rank in range(num_states):
            rank_state_dict = pickle.load(open(checkpoint_dir+'distributed_state_'+str(rank)+'.pkl', 'rb'))

            # collect optimizer states for later comparison since they are sharded
            aggregate_states(optimizer_states, rank_state_dict['optimizer'])

        # compare optimizer states
        """
        failure_case: non sharded to sharded weights fails
        for key, value in optimizer_states.items():
            assert_allclose(value.reshape(state_dict_pre_checkpoint['optimizer'][key].size()), state_dict_pre_checkpoint['optimizer'][key])
        """
    
    dist.barrier()
    os.remove(checkpoint_dir+'distributed_state_'+str(world_rank)+'.pkl')

@distributed_setup
def test_load_from_distributed_zero_full_precision_into_distributed_zero_full_precision(world_rank, world_size, device, checkpoint_dir = 'checkpoint_dir/distributed_zero/full_precision/'):
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
    
    # extract state dictionaries to compare
    state_dict_post_checkpoint, model = load_evaluate_extract_state_dict(device, orttrainer.ORTTrainerOptions(opts), checkpoint_dir)
    state_dict_post_checkpoint = split_state_dict(state_dict_post_checkpoint)

    state = pickle.load(open(checkpoint_dir+'state_dict_'+str(world_rank)+'.pkl', 'rb'))
    state_dict_pre_checkpoint = split_state_dict(state['state_dict_'+str(world_rank)])

    # compare all states
    # model states
    for key, value in state_dict_pre_checkpoint['fp32_param'].items():
        assert_allclose(value, state_dict_post_checkpoint['fp32_param'][key])

    # compare optimizer states
    for key, value in state_dict_pre_checkpoint['optimizer'].items():
        assert_allclose(value, state_dict_post_checkpoint['optimizer'][key])

@distributed_setup
def test_load_from_distributed_zero_mixed_precision_into_distributed_zero_full_precision(world_rank, world_size, device, checkpoint_dir = 'checkpoint_dir/distributed_zero/mixed_precision/'):
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
    
    # extract state dictionaries to compare
    state_dict_post_checkpoint, model = load_evaluate_extract_state_dict(device, orttrainer.ORTTrainerOptions(opts), checkpoint_dir)
    state_dict_post_checkpoint = split_state_dict(state_dict_post_checkpoint)

    state = pickle.load(open(checkpoint_dir+'state_dict_'+str(world_rank)+'.pkl', 'rb'))
    state_dict_pre_checkpoint = split_state_dict(state['state_dict_'+str(world_rank)])

    # compare all states
    # model states
    """
    failure_case: fp32 weights pre checkpoint are sharded. But since this is a one to one mapping (from distributed zero to distributed zero albeit
    mixed to full precision), the fp32 weights are not aggregated before copying into the new trainer
    for key, value in state_dict_pre_checkpoint['fp16_param'].items():
        assert_allclose(value, state_dict_post_checkpoint['fp32_param'][key[:-5]], atol=1e-3)
    """

    # compare optimizer states
    for key, value in state_dict_pre_checkpoint['optimizer'].items():
        assert_allclose(value, state_dict_post_checkpoint['optimizer'][key])

@distributed_setup
def test_load_from_distributed_zero_mixed_precision_into_distributed_zero_mixed_precision(world_rank, world_size, device, checkpoint_dir = 'checkpoint_dir/distributed_zero/mixed_precision/'):
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
    
    # extract state dictionaries to compare
    state_dict_post_checkpoint, model = load_evaluate_extract_state_dict(device, orttrainer.ORTTrainerOptions(opts), checkpoint_dir)
    state_dict_post_checkpoint = split_state_dict(state_dict_post_checkpoint)

    state = pickle.load(open(checkpoint_dir+'state_dict_'+str(world_rank)+'.pkl', 'rb'))
    state_dict_pre_checkpoint = split_state_dict(state['state_dict_'+str(world_rank)])

    # compare all states
    # model states
    for key, value in state_dict_pre_checkpoint['fp16_param'].items():
        assert_allclose(value, state_dict_post_checkpoint['fp16_param'][key])

    # compare optimizer states
    for key, value in state_dict_pre_checkpoint['optimizer'].items():
        assert_allclose(value, state_dict_post_checkpoint['optimizer'][key])

@distributed_setup
def test_load_from_distributed_zero_full_precision_into_distributed_zero_mixed_precision(world_rank, world_size, device, checkpoint_dir = 'checkpoint_dir/distributed_zero/full_precision/'):
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
    
    # extract state dictionaries to compare
    state_dict_post_checkpoint, model = load_evaluate_extract_state_dict(device, orttrainer.ORTTrainerOptions(opts), checkpoint_dir)
    state_dict_post_checkpoint = split_state_dict(state_dict_post_checkpoint)

    state = pickle.load(open(checkpoint_dir+'state_dict_'+str(world_rank)+'.pkl', 'rb'))
    state_dict_pre_checkpoint = split_state_dict(state['state_dict_'+str(world_rank)])

    # compare all states
    # model states
    for key, value in state_dict_post_checkpoint['fp16_param'].items():
        assert_allclose(value, state_dict_pre_checkpoint['fp32_param'][key[:-5]], atol=1e-3)

    # compare optimizer states
    for key, value in state_dict_pre_checkpoint['optimizer'].items():
        assert_allclose(value, state_dict_post_checkpoint['optimizer'][key])

if __name__ == '__main__':
    function_map = {
        # all config to single node config
        'test_load_from_single_node_full_precision_into_single_node_full_precision': test_load_from_single_node_full_precision_into_single_node_full_precision,
        'test_load_from_single_node_mixed_precision_into_single_node_mixed_precision': test_load_from_single_node_mixed_precision_into_single_node_mixed_precision,
        'test_load_from_single_node_mixed_precision_into_single_node_full_precision': test_load_from_single_node_mixed_precision_into_single_node_full_precision,
        'test_load_from_single_node_full_precision_into_single_node_mixed_precision': test_load_from_single_node_full_precision_into_single_node_mixed_precision,
        'test_load_from_data_parallelism_full_precision_into_single_node_full_precision': test_load_from_data_parallelism_full_precision_into_single_node_full_precision,
        'test_load_from_data_parallelism_mixed_precision_into_single_node_full_precision': test_load_from_data_parallelism_mixed_precision_into_single_node_full_precision,
        'test_load_from_data_parallelism_mixed_precision_into_single_node_mixed_precision': test_load_from_data_parallelism_mixed_precision_into_single_node_mixed_precision,
        'test_load_from_data_parallelism_full_precision_into_single_node_mixed_precision': test_load_from_data_parallelism_full_precision_into_single_node_mixed_precision,
        'test_load_from_distributed_zero_full_precision_into_single_node_full_precision': test_load_from_distributed_zero_full_precision_into_single_node_full_precision,
        'test_load_from_distributed_zero_mixed_precision_into_single_node_full_precision': test_load_from_distributed_zero_mixed_precision_into_single_node_full_precision,
        'test_load_from_distributed_zero_mixed_precision_into_single_node_mixed_precision': test_load_from_distributed_zero_mixed_precision_into_single_node_mixed_precision,
        'test_load_from_distributed_zero_full_precision_into_single_node_mixed_precision': test_load_from_distributed_zero_full_precision_into_single_node_mixed_precision,

        # all config to data parallel node config
        'test_load_from_single_node_full_precision_into_data_parallelism_full_precision': test_load_from_single_node_full_precision_into_data_parallelism_full_precision,
        'test_load_from_single_node_mixed_precision_into_data_parallelism_full_precision': test_load_from_single_node_mixed_precision_into_data_parallelism_full_precision,
        'test_load_from_single_node_mixed_precision_into_data_parallelism_mixed_precision': test_load_from_single_node_mixed_precision_into_data_parallelism_mixed_precision,
        'test_load_from_single_node_full_precision_into_data_parallelism_mixed_precision': test_load_from_single_node_full_precision_into_data_parallelism_mixed_precision,
        'test_load_from_data_parallelism_full_precision_into_data_parallelism_full_precision': test_load_from_data_parallelism_full_precision_into_data_parallelism_full_precision,
        'test_load_from_data_parallelism_mixed_precision_into_data_parallelism_full_precision': test_load_from_data_parallelism_mixed_precision_into_data_parallelism_full_precision,
        'test_load_from_data_parallelism_mixed_precision_into_data_parallelism_mixed_precision': test_load_from_data_parallelism_mixed_precision_into_data_parallelism_mixed_precision,
        'test_load_from_data_parallelism_full_precision_into_data_parallelism_mixed_precision': test_load_from_data_parallelism_full_precision_into_data_parallelism_mixed_precision,
        'test_load_from_distributed_zero_full_precision_into_data_parallelism_full_precision': test_load_from_distributed_zero_full_precision_into_data_parallelism_full_precision,
        'test_load_from_distributed_zero_mixed_precision_into_data_parallelism_full_precision': test_load_from_distributed_zero_mixed_precision_into_data_parallelism_full_precision,
        'test_load_from_distributed_zero_mixed_precision_into_data_parallelism_mixed_precision': test_load_from_distributed_zero_mixed_precision_into_data_parallelism_mixed_precision,
        'test_load_from_distributed_zero_full_precision_into_data_parallelism_mixed_precision': test_load_from_distributed_zero_full_precision_into_data_parallelism_mixed_precision,

        # all config to distributed zero node config
        'test_load_from_single_node_full_precision_into_distributed_zero_full_precision': test_load_from_single_node_full_precision_into_distributed_zero_full_precision,
        'test_load_from_single_node_mixed_precision_into_distributed_zero_full_precision': test_load_from_single_node_mixed_precision_into_distributed_zero_full_precision,
        'test_load_from_single_node_mixed_precision_into_distributed_zero_mixed_precision': test_load_from_single_node_mixed_precision_into_distributed_zero_mixed_precision,
        'test_load_from_single_node_full_precision_into_distributed_zero_mixed_precision': test_load_from_single_node_full_precision_into_distributed_zero_mixed_precision,
        'test_load_from_data_parallelism_full_precision_into_distributed_zero_full_precision': test_load_from_data_parallelism_full_precision_into_distributed_zero_full_precision,
        'test_load_from_data_parallelism_mixed_precision_into_distributed_zero_full_precision': test_load_from_data_parallelism_mixed_precision_into_distributed_zero_full_precision,
        'test_load_from_data_parallelism_mixed_precision_into_distributed_zero_mixed_precision': test_load_from_data_parallelism_mixed_precision_into_distributed_zero_mixed_precision,
        'test_load_from_data_parallelism_full_precision_into_distributed_zero_mixed_precision': test_load_from_data_parallelism_full_precision_into_distributed_zero_mixed_precision,
        'test_load_from_distributed_zero_full_precision_into_distributed_zero_full_precision': test_load_from_distributed_zero_full_precision_into_distributed_zero_full_precision,
        'test_load_from_distributed_zero_mixed_precision_into_distributed_zero_full_precision': test_load_from_distributed_zero_mixed_precision_into_distributed_zero_full_precision,
        'test_load_from_distributed_zero_mixed_precision_into_distributed_zero_mixed_precision': test_load_from_distributed_zero_mixed_precision_into_distributed_zero_mixed_precision,
        'test_load_from_distributed_zero_full_precision_into_distributed_zero_mixed_precision': test_load_from_distributed_zero_full_precision_into_distributed_zero_mixed_precision
    }
    parser = argparse.ArgumentParser(description='Test saved states of trainers to loaded states')
    parser.add_argument('scenario', choices=function_map.keys(), help='training scenario to test saved and loaded states')
    args = parser.parse_args()
    function_map[args.scenario]()