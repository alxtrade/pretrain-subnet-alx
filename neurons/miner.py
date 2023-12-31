# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Copyright © 2023 const

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import os
import wandb
import torch
import string
import random
import argparse
import pretrain
import bittensor as bt

# === Config ===
def get_config():
    """
    Set up and parse the command-line arguments to configure the system.
    Returns:
        A namespace object containing the configuration parameters.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=False, help="Override model path")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device name.")
    parser.add_argument("--load_best", action='store_true', help='If set, the miner loads the best model from wandb to train off.')
    parser.add_argument("--load_run_id", type=str, default=None, help='If passed loads the model under this run id')
    parser.add_argument("--continue_id", type=str, default=None, help='If passed continues from the model on the passed run.')
    parser.add_argument("--num_epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.00001, help="Learning rate.")
    parser.add_argument("--bs", type=int, default=2, help="Batch size")
    parser.add_argument("--sl", type=int, default=512, help="Sequence length")
    parser.add_argument("--accumulation_steps", type=int, default=10, help="The number of training accumulation steps.")
    parser.add_argument("--pages_per_epoch", type=int, default=10, help="Number of pages trained on per epoch")

    bt.wallet.add_args(parser)
    bt.subtensor.add_args(parser)
    bt.logging.add_args(parser)

    config = bt.config(parser)
    config.full_path = os.path.expanduser(
        "{}/{}/{}/netuid{}/{}".format(
            config.logging.logging_dir,
            config.wallet.name,
            config.wallet.hotkey,
            pretrain.NETUID,
            "miner",
        )
    )

    if config.model_path is None:
        config.model_path = config.full_path + '/' + 'model.pth'

    if not os.path.exists(os.path.dirname(config.model_path)):
        os.makedirs(os.path.dirname(config.model_path), exist_ok=True)

    return config

config = get_config()

print(config)

bt.logging(config=config)
wallet = bt.wallet(config=config)
subtensor = bt.subtensor(config=config)
metagraph = subtensor.metagraph(pretrain.NETUID)

if wallet.hotkey.ss58_address not in metagraph.hotkeys:
    bt.logging.error(f"You are not registered. Use `btcli s register --netuid {pretrain.NETUID}` to register.")
    exit()

my_uid = metagraph.hotkeys.index(wallet.hotkey.ss58_address)
bt.logging.success(f'You are registered with address: {wallet.hotkey.ss58_address} and uid: {my_uid}')

model = pretrain.model.get_model()
torch.save(model.state_dict(), config.model_path)
api = wandb.Api(timeout=100)

def get_run_from_id(run_id):
    run_path = f"opentensor-dev/{pretrain.WANDB_PROJECT}/{run_id}"
    bt.logging.success(f'Loading model from path: {run_path}')
    return api.run(run_path)

def load_model_from_run(run):
    model_file = run.file("model.pth")
    model_file.download(replace=True, root=os.path.dirname(config.model_path))
    bt.logging.success(f'Loaded and saved model to: {config.model_path}')

if config.load_run_id is not None:
    bt.logging.success(f'Loading based on --config.load_run_id {config.model_path}')
    load_model_from_run(get_run_from_id(config.load_run_id))

elif config.load_best:
    bt.logging.success(f'Loading based on --config.load_best')
    best_uid = max(range(256), key=lambda uid: metagraph.I[uid].item())
    print(f"best uid is {best_uid}")
    runs = api.runs(
        f"opentensor-dev/{pretrain.WANDB_PROJECT}",
        filters={
            "config.version": pretrain.__version__,
            "config.type": "miner",
            "config.run_name": {
                "$regex": f"miner-{best_uid}-.*"
            }
        }
    )
    load_model_from_run(get_run_from_id(runs[0].id))

elif config.continue_id:
    run = get_run_from_id(config.continue_id)
    run_hotkey = run.config['hotkey']
    load_model_from_run(run)

else:
    bt.logging.success(f'Starting model from scratch')

model_weights = torch.load(config.model_path, map_location=torch.device(config.device))
model.load_state_dict(model_weights)
model.zero_grad()
model.train()
model.to(config.device)

optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=0.01)

import random

best_avg_loss = float('inf')

import json
run_id_file = config.full_path + '/run.json'
try:
    with open(run_id_file, 'r') as f:
        run_id = json.load(f)['WANDB_RUN_ID']
        bt.logging.success(f'Continuing run, loaded run_id: {run_id}')
except Exception as e:
    run_id = wandb.util.generate_id()
    bt.logging.success(f'First run, creating new run_id: {run_id} {e}')

with open(run_id_file, 'w') as f:
    json.dump({'WANDB_RUN_ID': run_id}, f)
    bt.logging.success(f'Saved: {run_id} to file.')

run_name = f'miner-{my_uid}-' + ''.join(random.choice(string.ascii_uppercase + string.digits) for i in range(10))
config.uid = my_uid
config.hotkey = wallet.hotkey.ss58_address
config.run_name = run_name
config.version = pretrain.__version__
config.type = 'miner'
wandb_run = wandb.init(
    id=run_id,
    name=run_name,
    anonymous="allow",
    project=pretrain.WANDB_PROJECT,
    entity='opentensor-dev',
    config=config,
    dir=config.full_path,
    allow_val_change=True,
)

signature = wallet.hotkey.sign(wandb_run.id.encode()).hex()
config.signature = signature
wandb.config.update(config, allow_val_change=True)
bt.logging.success(f'Successfully signed wandb run with signature {config.signature}')

wandb.save(config.model_path)
bt.logging.success('Pushed artifact to the wandb run.')

epoch_step = 0
global_step = 0
n_acc_steps = 0
accumulation_steps = config.accumulation_steps

try:
    while epoch_step < config.num_epochs or config.num_epochs == -1:
        epoch_loss = 0.0
        bt.logging.success(f"Loading {config.pages_per_epoch} pages for training this epoch")
        random_pages = [random.randint(1, pretrain.dataset.SubsetFalconLoader.max_pages) for _ in range(config.pages_per_epoch)]
        loader = pretrain.dataset.SubsetFalconLoader(
            batch_size=config.bs,
            sequence_length=config.sl,
            pages=random_pages
        )

        n_batches = 0
        optimizer.zero_grad()

        for i, batch in enumerate(loader):
            inputs = batch.to(model.device)
            outputs = model(inputs, labels=inputs)

            loss = outputs.loss / accumulation_steps
            loss.backward()

            if (i + 1) % accumulation_steps == 0:
                n_acc_steps += 1
                optimizer.step()
                optimizer.zero_grad()
                bt.logging.success(f'Step: {n_acc_steps} loss: {outputs.loss.detach().item()}')
                wandb.log({'loss': outputs.loss.detach(), 'n_batches': n_batches}, step=n_acc_steps)

            torch.cuda.empty_cache()

            n_batches += 1
            global_step += 1
            epoch_loss += outputs.loss.detach().item()

        avg_loss = epoch_loss / n_batches
        bt.logging.success(f'Epoch: {epoch_step} average loss: {avg_loss}')
        epoch_step += 1

        if avg_loss < best_avg_loss * (1 - pretrain.timestamp_epsilon):
            best_avg_loss = avg_loss
            bt.logging.success(f'New best average loss: {best_avg_loss}. Saving model...')

            torch.save(model.state_dict(), config.model_path)
            wandb.save(config.model_path)
            bt.logging.success('Pushed the new artifact to the wandb run.')

finally:
    wandb.finish()
