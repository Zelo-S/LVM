import glob
import time
import natsort
from datetime import datetime
import gc
import pathlib
import os
import json
from PIL import Image
import numpy as np
import mlxu
from tqdm import tqdm, trange
from multiprocessing import Pool, set_start_method
import einops
import torch
from pathlib import Path
from torch.nn import CrossEntropyLoss

import matplotlib.pyplot as plt

from inference import MultiProcessInferenceModel, LocalInferenceModel
from utils import read_image_to_tensor, MultiProcessImageSaver
from BridgeDataset_mmap_M import BridgeDatasetV2
from torch.utils.data import DataLoader

import inspect2
LOG = False
def log(str_):
    f = inspect2.currentframe()
    i = inspect2.getframeinfo(f.f_back)
    if LOG: print(f"{i.filename}:{i.lineno} --> {i.function} -> {str_}")

FLAGS, _ = mlxu.define_flags_with_default(
    input_file='./input.json', # not used rn
    checkpoint='../LVM_ckpts',
    input_base_dir='',
    output_base_dir='',
    evaluate_mse=False,
    json_input_key='input',
    json_output_key='output',
    json_target_key='target',
    n_new_frames=1,
    n_candidates=1,
    context_frames=16,
    temperature=1.0,
    top_p=1.0,
    n_workers=1,
    dtype='float16',
    torch_devices='',
    batch_size_factor=1,
    max_examples=0,
    resize_output='original',
    include_input=False,
)

# create this according to the json file.
class MultiFrameDataset(torch.utils.data.Dataset):
    def __init__(self, input_files, output_files, target_files=None):
        assert len(input_files)
        self.input_files = input_files
        self.output_files = output_files
        self.target_files = target_files

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, idx):
        original_size = Image.open(self.input_files[idx][-1]).size
        input_images = np.stack(
            [read_image_to_tensor(f) for f in self.input_files[idx]],
            axis=0
        )

        if self.target_files is not None:
            target_images = np.stack(
                [read_image_to_tensor(f) for f in self.target_files[idx]],
                axis=0
            )
        else:
            target_images = None
        return input_images, target_images, self.output_files[idx], np.array(original_size)


def main(_):
    start_time = time.time()
    # log(f"Clearning memory: {torch.cuda.empty_cache()} {gc.collect()}")

    assert FLAGS.checkpoint != ''
    set_start_method('spawn')
    print(f'Loading checkpoint from {FLAGS.checkpoint}')
    print(f'Evaluating input file from {FLAGS.input_file}')
    log(f"Dtype: {FLAGS.dtype}")
    # build a model.
    model = LocalInferenceModel(
        checkpoint=FLAGS.checkpoint,
        dtype=FLAGS.dtype,
        context_frames=FLAGS.context_frames,
        use_lock=True,
    )
    
    log(f"Loaded model in: {time.time() - start_time}")
    log(f"Memory usage post-model: {torch.cuda.memory_allocated()}")

    # input_files: the json file that needs to be generated by the other file.
    input_files = []
    output_files = []

    if FLAGS.evaluate_mse:
        target_files = []
    else:
        target_files = None

    with mlxu.open_file(FLAGS.input_file, 'r') as f:
        for line in f:
            record = json.loads(line)
            input_files.append(record[FLAGS.json_input_key])
            output_files.append(record[FLAGS.json_output_key])
            if FLAGS.evaluate_mse:
                target_files.append(record[FLAGS.json_target_key])

    if FLAGS.max_examples > 0:
        input_files = input_files[:FLAGS.max_examples]
        output_files = output_files[:FLAGS.max_examples]
        if FLAGS.evaluate_mse:
            target_files = target_files[:FLAGS.max_examples]

    if FLAGS.input_base_dir != '':
        input_files = [
            [os.path.join(FLAGS.input_base_dir, x) for x in y]
            for y in input_files
        ]
        if FLAGS.evaluate_mse:
            target_files = [
                [os.path.join(FLAGS.input_base_dir, x) for x in y]
                for y in target_files
            ]

    if FLAGS.output_base_dir != '':
        os.makedirs(FLAGS.output_base_dir, exist_ok=True)
        output_files = [
            os.path.join(FLAGS.output_base_dir, x)
            for x in output_files
        ]
    
    data_dir = "/scratch/prachig3/BridgeDataV2_scripted_numpy_256_memmap_trajlen49/"
    
    # print(f"Glob: {glob.glob(os.path.join(data_dir, '**', 'val', 'out.npy'), recursive=True)}")
    val_dataset = BridgeDatasetV2(
        data_dir=data_dir,
        sequence_length=8,
        train=False,
        max_blocks=8,
        augment=False,
        load_language=False,
        trajectory_sampling_strategy='many',
        shuffle_sampled_sequence=False
    ) # trying to keep validation comparable

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        prefetch_factor=2,
        pin_memory=True
    )
    print("Finished loading dataset")

    log(f"Memory usage post-data-loader: {torch.cuda.memory_allocated()}")
    
    for batch_triplet, batch_targets in tqdm(val_loader, ncols=0):
        obs = batch_triplet["observations"]
        # next_obs = batch_triplet["next_observations"]
        # actions = batch_triplet["actions"]
        # print(f"Batch Targets shape: {batch_targets.keys()}")
        target_obs = batch_targets["observations"]
        
        # batch_images is input.
        batch_images = obs
        batch_images = batch_images.numpy()
        log(f"Batch Image Shape: {batch_images.shape}") # TODO: expects (b, seq, c, h, w)
        
        target_images = target_obs.numpy()
        log(f"Target Image Shape: {target_images.shape}") # 
        
        context_length = batch_images.shape[1]
        
        generated_token_stack = []
        groundtruth_token_stack = []

        for i in range(context_length+1):
            log(f"==== Generating images UP TO: {i} ====")
            if i == 0:
                continue
            generated_images, input_tokenized, generated_tokens = model(
                batch_images[:, :i, :, :, :],
                FLAGS.n_new_frames,
                FLAGS.n_candidates,
                temperature=FLAGS.temperature,
                top_p=FLAGS.top_p
            )
            generated_image = np.array(generated_images)
            input_tokenized = input_tokenized[0].cpu().numpy()
            generated_tokens = generated_tokens[0].cpu().numpy()
            log(f"Generated image: {generated_image.shape}") # TODO: Generated image: (1, n_candidates, H, W, C)
            log(f"Tokenized Input images: {input_tokenized.shape}") # TODO: Tokenized Input: (1, 256 * i)
            log(f"Generated Tokens: {generated_tokens.shape}") # TODO: Generated Tokens: (1, 256 * (i+1))
            
            generated_token_stack.append(generated_tokens[:, -256:])
            groundtruth_token_stack = input_tokenized

        generated_token_stack = np.asarray(generated_token_stack)
        generated_token_stack = generated_token_stack.reshape(1, -1)

        print(f"Generated token stack: {generated_token_stack.shape}")
        print(f"GT token stack: {groundtruth_token_stack.shape}")
        loss = CrossEntropyLoss()
        print(f"generated_tokens: {generated_token_stack}") 
        print(f"GT_tokens: {groundtruth_token_stack}") 
        target_tens = torch.tensor(groundtruth_token_stack[:, :-1])
        input_tens = torch.tensor(generated_token_stack[:, 1:])
        cle = loss(input_tens, target_tens)
        print(f"==> Cle is: {cle}")

    """
    root_dir = "/home/stevex2/data/vision-data/bridge_v2/bin_obj/im1/"
    directory = pathlib.Path(root_dir)
    input_file_names = [natsort.natsorted([root_dir + f.name for f in directory.iterdir() if f.is_file()])]
    input_files[0] = input_files[0][1]
    dataset = MultiFrameDataset(input_file_names, output_files, target_files)


    input_ims, _, output_im, og_size = dataset.__getitem__(0)
    log(input_ims.shape)
    log(output_im)
    log(og_size)
    
    input_ims = np.expand_dims(input_ims, axis=0)
    log(f"Post expansion input im shape: {input_ims.shape}")

    context_length = input_ims.shape[1]
    log(f"Ctx Length: {context_length}")
    
    
    generated_images = model(
        input_ims,
        FLAGS.n_new_frames,
        FLAGS.n_candidates,
        temperature=FLAGS.temperature,
        top_p=FLAGS.top_p
    )
    
    log(f"\nCuda usage: {torch.cuda.utilization()}\n")
    
    log(f"Generated Images Shape: {len(generated_images)} {generated_images[0][0].shape}")

    im = input_ims[0][-1]
    log(f"Original Im has shape: {np.asarray(im).shape}")
    
    print(len(generated_images[0]))
    for candidate in generated_images[0]:
        print("Candidate:", np.asarray(candidate).shape)
        im_out = Image.fromarray((candidate[0] * 255).astype(np.uint8))
        log(f"Output Image shape: {(candidate[0] * 255).astype(np.uint8).shape}")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = root_dir + "output/"
        os.makedirs(save_dir, exist_ok=True)
        im_out.save(save_dir + f"output_image_{timestamp}.jpg")
        time.sleep(1.1)
        print("Saving image")

    log(f"Took time: {time.time() - start_time}")
    """

if __name__ == "__main__":
    mlxu.run(main)