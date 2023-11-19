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

def setup_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=False, help="Override model path")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device name.")
    parser.add_argument("--load_best", action='store_true', help='If set, the miner loads the best model from wandb to train off.')
    # ... (other arguments)

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

    config.model_path = config.model_path or (config.full_path + '/' + 'model.pth')

    if not os.path.exists(os.path.dirname(config.model_path)):
        os.makedirs(os.path.dirname(config.model_path), exist_ok=True)

    # Adjust hyperparameters for training
    config.lr = 0.0001  # Adjust learning rate
    config.bs = 32  # Adjust batch size
    config.sl = 128  # Adjust sequence length
    config.accumulation_steps = 1  # Adjust accumulation steps
    config.pages_per_epoch = 5  # Adjust the number of pages trained on per epoch

    return config

def load_model_from_run(api, run_id, model_path):
    run_path = f"opentensor-dev/{pretrain.WANDB_PROJECT}/{run_id}"
    run = api.run(run_path)
    model_file = run.file("model.pth")
    model_file.download(replace=True, root=os.path.dirname(model_path))
    bt.logging.success(f'Loaded and saved model to: {model_path}')

def load_model(config, api):
    if config.load_run_id:
        bt.logging.success(f'Loading based on --config.load_run_id {config.model_path}')
        load_model_from_run(api, config.load_run_id, config.model_path)
    elif config.load_best:
        bt.logging.success(f'Loading based on --config.load_best')
        best_uid = max(range(256), key=lambda uid: config.metagraph.I[uid].item())
        runs = api.runs(
            f"opentensor-dev/{pretrain.WANDB_PROJECT}",
            filters={
                "config.version": pretrain.__version__,
                "config.type": "miner",
                "config.run_name": {"$regex": f"miner-{best_uid}-.*"}
            }
        )
        load_model_from_run(api, runs[0].id, config.model_path)
    elif config.continue_id:
        run = api.run(f"opentensor-dev/{pretrain.WANDB_PROJECT}/{config.continue_id}")
        run_hotkey = run.config['hotkey']
        load_model_from_run(api, config.continue_id, config.model_path)
    else:
        bt.logging.success('Starting model from scratch')

def setup_wandb(config):
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

    run_name = f'miner-{config.uid}-' + ''.join(random.choice(string.ascii_uppercase + string.digits) for i in range(10))
    config.run_name = run_name
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

    signature = config.wallet.hotkey.sign(wandb_run.id.encode()).hex()
    config.signature = signature
    wandb.config.update(config, allow_val_change=True)
    bt.logging.success(f'Successfully signed wandb run with signature {config.signature}')

def train_model(model, optimizer, loader, config, wandb_run):
    # ... (your existing training loop code)

def main():
    try:
        config = setup_config()
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

        load_model(config, api)

        model_weights = torch.load(config.model_path, map_location=torch.device(config.device))
        model.load_state_dict(model_weights)
        model.zero_grad()
        model.train()
        model.to(config.device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=0.01)

        wandb.save(config.model_path)
        bt.logging.success('Pushed artifact to the wandb run.')

        train_model(model, optimizer, loader, config, wandb_run)

    except Exception as e:
        # ... (add more comprehensive error handling)

    finally:
        wandb.finish()

if __name__ == "__main__":
    main()