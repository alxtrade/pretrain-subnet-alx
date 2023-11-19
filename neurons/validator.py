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
import json
import math
import time
import wandb
import torch
import typing
import string
import random
import asyncio
import argparse
import pretrain
import traceback
import threading
import bittensor as bt
from tqdm import tqdm
from typing import Dict, List
from rich.table import Table
from rich.console import Console

# Global artifact name
UPDATE_TIMEOUT = 60*60*2
ARTIFACT_NAME:str = "model.pth"
RUN_STEP_MAX_TIME = 60 * 20 # 20 min run step timeout.
class Validator:

    @staticmethod
    def config():
        parser = argparse.ArgumentParser()
        parser.add_argument( "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device name.")
        parser.add_argument( '--wandb.off', dest = 'wandb.on', action='store_false', help='Turn off wandb logging.' )
        parser.add_argument( '--blocks_per_epoch', type=int, default=360, help='Number of blocks to wait before setting weights.' )
        parser.add_argument( '--pages_per_eval', type=int, default=3, help='Number of pages used to eval each step.' )
        parser.add_argument( '--sample_min', type=int, default=10, help='Number of uids to eval each step.' )
        parser.add_argument( '--reset_wandb', action='store_true', help='Creates a new wandb run instead of using an older on.' )
        parser.add_argument( '--dont_set_weights', action='store_true', help='Creates a new wandb run instead of using an older on.' )
        bt.subtensor.add_args(parser)
        bt.logging.add_args(parser)
        bt.wallet.add_args(parser)
        bt.axon.add_args(parser)
        config = bt.config(parser)
        config.full_path = os.path.expanduser(
            "{}/{}/{}/netuid{}/{}".format(
                config.logging.logging_dir,
                config.wallet.name,
                config.wallet.hotkey,
                pretrain.NETUID,
                "validator",
            )
        )
        if not os.path.exists(config.full_path):
            os.makedirs(config.full_path, exist_ok=True)
        return config

    def init_wandb(self):
        if self.config.wandb.on:
            import json
            run_id_file = self.config.full_path + '/run.json'
            try:
                if self.config.reset_wandb: raise Exception('go to create new run.')
                with open( run_id_file, 'r' ) as f:
                    self.run_id = json.load( f )['WANDB_RUN_ID']
                    bt.logging.success(f'Continuing run, loaded run_id: {self.run_id}')
            except Exception as e: 
                self.run_id = wandb.util.generate_id()
                bt.logging.success(f'First run, creating new run_id: {self.run_id} {e}')
            with open( run_id_file, 'w' ) as f:
                json.dump({'WANDB_RUN_ID': self.run_id}, f)
                bt.logging.success(f'Saved: {self.run_id} to file.')
            self.run_name = f'validator-{self.uid}-' + ''.join(random.choice( string.ascii_uppercase + string.digits ) for i in range(10))
            self.config.uid = self.uid
            self.config.hotkey = self.wallet.hotkey.ss58_address
            self.config.run_name = self.run_name
            self.config.type = "validator"
            self.config.version = pretrain.__version__
            self.wandb_run = wandb.init(
                id = self.run_id,
                name = self.run_name,
                anonymous = "allow",
                reinit = False,
                project = pretrain.WANDB_PROJECT,
                entity = 'opentensor-dev',
                config = self.config,
                dir = self.config.full_path,
            )
            self.config.signature = self.wallet.hotkey.sign( self.wandb_run.id.encode() ).hex()
            wandb.config.update( self.config, allow_val_change=True )

    def __init__(self ):
        self.config = Validator.config()
        bt.logging( config = self.config )

        # === Bittensor objects ====
        self.wallet = bt.wallet( config = self.config )
        self.subtensor = bt.subtensor( config = self.config )
        self.dendrite = bt.dendrite( wallet = self.wallet )
        self.metagraph = self.subtensor.metagraph( pretrain.NETUID )
        torch.backends.cudnn.benchmark = True
        if self.wallet.hotkey.ss58_address not in self.metagraph.hotkeys: raise Exception(f"You are not registered. Use `btcli s register --netuid {pretrain.NETUID}` to register.")
        self.uid = self.metagraph.hotkeys.index( self.wallet.hotkey.ss58_address )
        bt.logging.success( f'You are registered with address: {self.wallet.hotkey.ss58_address} and uid: {self.uid}' )
        self.init_wandb()

        # === Running args ===
        self.weights = torch.zeros_like(self.metagraph.S)
        self.epoch_step = 0 
        self.global_step = 0
        self.last_epoch = self.metagraph.block.item()
        self.last_update_check = {}
        self.metadata = { uid: pretrain.utils.load_metadata_for_uid( uid ) for uid in self.metagraph.uids.tolist() }
        self.uids_to_eval = set()
        for uid in self.metagraph.uids.tolist():
            if self.metadata[uid] != None:
                self.uids_to_eval.add( uid )

        # == Initialize the update thread ==
        self.stop_event = threading.Event()
        self.update_thread = threading.Thread(target=self.update_models, daemon=True)
        self.update_thread.start()

    def __del__(self):
        if hasattr( self, 'stop_event'):
            self.stop_event.set()
            self.update_thread.join()

    def update_models( self ):
        # The below loop iterates across all miner uids and checks to see 
        # if they should be updated.
        last_uid_update = -1
        while not self.stop_event.is_set():
            if self.stop_event.is_set(): return
            block = self.subtensor.block 
            uid = block % 256
            if uid == last_uid_update: 
                time.sleep(1)
                continue
            last_uid_update = uid
            bt.logging.success( f'Updating model under uid: {uid} for block: {block}')
            if pretrain.utils.update_model_for_uid( uid, self.metagraph ):
                self.uids_to_eval.add( uid )
                bt.logging.trace(f'uids to eval add: {uid}')

    def compute_losses_per_page( self, uid, batches_per_page: Dict[int, List[torch.Tensor]] ) -> Dict[int, List[float]]:
        try:
            # Load the pre-trained model from the specified path
            model_path = self.metadata[uid]['model_path']
            model = pretrain.model.get_model()
            model_weights = torch.load(model_path, map_location=torch.device(self.config.device))
            model.load_state_dict(model_weights)
            model.eval()  # Set the model to evaluation mode
            model.to(self.config.device)  # Move the model to the appropriate device
        except Exception as e:
            bt.logging.debug(f"Error loading {uid} model with error: {e}")
            inf_losses = {}
            for page, batches in batches_per_page.items():
                inf_losses[page] = [math.inf for _ in batches]
            return inf_losses

        # Initialize a dictionary to store loss values for each page
        losses_per_page = {}

        # Iterate over each page and its corresponding batches
        for page, batches in batches_per_page.items():
            page_losses = []  # List to store losses for the current page

            # Process each batch and compute its loss
            for batch in batches:
                try:
                    # Perform a forward pass with the model to compute the loss
                    inputs = batch.to(self.config.device)
                    outputs = model(inputs, labels=inputs)
                    loss = outputs.loss.item()  # Get the scalar loss value
                    page_losses.append(loss)
                except Exception as e:
                    # Log the exception and append infinity to indicate failure
                    bt.logging.error(f"Exception occurred: {e}")
                    traceback.print_exc()  # Correctly print the stack trace
                    page_losses.append(math.inf)
            
            # Update the dictionary with the losses for the current page
            losses_per_page[page] = page_losses

        return losses_per_page

    async def try_set_weights( self, ttl: int ):
        async def _try_set_weights():
            try:
                self.weights.nan_to_num( 0.0 )
                self.subtensor.set_weights(
                    netuid = pretrain.NETUID,
                    wallet = self.wallet,
                    uids = self.metagraph.uids,
                    weights = self.weights,
                    wait_for_inclusion=False,
                )
            except: pass
            ws, ui = self.weights.topk( len( self.weights ) )
            table = Table(title="All Weights")
            table.add_column("uid", justify="right", style="cyan", no_wrap=True)
            table.add_column("weight", style="magenta")
            for index, weight in list(zip(ui.tolist(), ws.tolist())):
                table.add_row(str(index), str(round(weight, 4)))
            console = Console()
            console.print(table)
        try: await asyncio.wait_for( _try_set_weights() , ttl )
        except asyncio.TimeoutError: 
            bt.logging.error(f'Failed to set weights after {ttl} seconds')

    async def try_sync_metagraph( self, ttl: int ):
        async def _try_sync_metagraph():
            self.metagraph = self.subtensor.metagraph( pretrain.NETUID )
        try: await asyncio.wait_for( _try_sync_metagraph() , ttl )
        except asyncio.TimeoutError: 
            bt.logging.error(f'Failed to sync metagraph after {ttl} seconds')

    async def try_run_step( self, ttl: int ):
        async def _try_run_step():
            await self.run_step()
        try: await asyncio.wait_for( _try_run_step() , ttl )
        except asyncio.TimeoutError: 
            bt.logging.error(f'Failed to run step after {ttl} seconds')

    # Add a 20 minute max timeout.
    async def run_step( self ):
        # Load metadata.
        self.metadata = { uid: pretrain.utils.load_metadata_for_uid( uid ) for uid in self.metagraph.uids.tolist() }

        # Uids to eval is based on global uids to eval. Uids to eval is pruned every step
        # based on the best 50% of miners with replacement when those model are updated during the
        # update thread. uid to eval is bounded below at sample_min.
        uids = []
        for uid in list( self.uids_to_eval ):
            if self.metadata[uid] == None: continue
            if pretrain.utils.check_run_exists( uid, self.metadata[uid], self.metagraph ):
                uids.append( uid )
        random.shuffle( uids )
        bt.logging.success( f'Runnning step with uids: {uids}')

        # Generate random pages for evaluation and prepare batches for each page
        # the dataset contains >900 million pages to eval over.
        pages = [random.randint(1, pretrain.dataset.SubsetFalconLoader.max_pages) for _ in range(self.config.pages_per_eval)]
        batches_per_page = {
            page: list(pretrain.dataset.SubsetFalconLoader(
                batch_size = pretrain.batch_size,
                sequence_length = pretrain.sequence_length,
                pages = [page]
            )) for page in pages
        }

        # Compute losses per page and average losses for each uid.
        bt.logging.debug(f"computing losses on {uids}")
        best_average_loss = math.inf
        best_average_loss_uid = None
        losses_per_page_per_uid = { uid: None for uid in uids }
        average_loss_per_uid_per_page = { uid: {} for uid in uids }
        for uid_i in uids:
            # Compute losses across each batch of each page.
            losses = self.compute_losses_per_page( uid_i, batches_per_page )
            losses_per_page_per_uid[ uid_i ] = losses

            # Compute average loss per page
            for page_j in pages:
                page_losses = losses[ page_j ]
                average_loss = sum(page_losses)/len(page_losses)
                average_loss_per_uid_per_page[ uid_i ][ page_j ] = average_loss

            # Compute best average loss.
            total_average_loss = sum([ average_loss_per_uid_per_page[uid_i][page_j] for page_j in pages ] ) / len( pages )
            if total_average_loss < best_average_loss:
                best_average_loss = total_average_loss
                best_average_loss_uid = uid_i
    
            bt.logging.debug( f'Computed loss for uid: {uid_i} with total average loss: {total_average_loss}')

        # Win function.
        # Determines the winner based on the epsilon adjusted loss
        # Models that were created earlier have a 3% decrease in loss
        def better( i, j, p, b ):
            il = losses_per_page_per_uid[ i ][ p ][ b ]
            jl = losses_per_page_per_uid[ j ][ p ][ b ]
            if 'timestamp' not in self.metadata[ i ]: return False 
            if 'timestamp' not in self.metadata[ j ]: return True 
            it = self.metadata[ i ]['timestamp']
            jt = self.metadata[ j ]['timestamp']
            il = (1 - pretrain.timestamp_epsilon) * il if it < jt else il
            jl = (1 - pretrain.timestamp_epsilon) * jl if jt < it else jl
            if il < jl: return True
            else: return False

        # Compute wins this step.
        wins = { uid: 0 for uid in uids }
        win_rate = { uid: 0 for uid in uids }
        for i in uids:
            total_matches = 0
            for j in uids:
                if i == j: continue
                for p in pages:
                    for b, _ in enumerate( batches_per_page[ p ] ):
                        wins[ i ] += 1 if better( i, j, p, b ) else 0
                        total_matches += 1
            win_rate[ i ] = wins[ i ] / total_matches
   
        # Compute and update weights which is a temperatured softmax over wins.
        step_weights = torch.tensor([ win_rate[ uid ] for uid in uids ], dtype=torch.float32)
        softmax_step_weights = torch.softmax( step_weights / pretrain.temperature, dim=0 )
        new_weights = self.weights.clone()
        for i, uid in enumerate( uids ):
            new_weights[ uid ] = softmax_step_weights[ i ] 
        new_weights.nan_to_num( 0.0 )
        new_weights /= new_weights.sum()
        new_weights.nan_to_num( 0.0 )
        self.weights = pretrain.alpha * self.weights + ( 1 - pretrain.alpha ) * new_weights
        self.weights.nan_to_num( 0.0 )

        # Blacklist bad miners. Here we remove uids from eval set 
        # based on their win rate, this prunes miners down the sample min
        # miner uids are replaced when their model is updated on wandb after a timeout.
        removed = 0
        size = len( list(self.uids_to_eval) )
        for uid in uids:
            if size - removed <= self.config.sample_min: break
            if win_rate[uid] < 0.5:
                self.uids_to_eval.remove( uid )

        # Build step log
        step_log = {
            'timestamp': time.time(),
            'pages': pages,
            'uids': uids,
            'best_average_loss': best_average_loss,
            'best_average_loss_uid': best_average_loss_uid,
            'uid_data': {}
        }
        for uid in uids:
            try:
                average_losses = [average_loss_per_uid_per_page[uid][pagek] for pagek in pages]
                average_loss = sum(average_losses) / len(average_losses)
                step_log['uid_data'][ str(uid) ] = {
                    'uid': uid,
                    'runid': self.metadata[ uid ]['runid'],
                    'timestamp': self.metadata[ uid ]['timestamp'],
                    'last_update': self.metadata[ uid ]['last_update'],
                    'blacklisted': self.metadata[ uid ]['blacklisted'],
                    'average_losses': average_losses,
                    'average_loss': average_loss,
                    'win_rate': win_rate[ uid ],
                    'win_total': wins[ uid ],
                    'weight': self.weights[ uid ].item()
                }
            except:
                continue
        table = Table(title="Step")
        table.add_column("uid", justify="right", style="cyan", no_wrap=True)
        table.add_column("average_loss", style="magenta")
        table.add_column("win_rate", style="magenta")
        table.add_column("win_total", style="magenta")
        table.add_column("weights", style="magenta")
        table.add_column("last_update", style="magenta")
        table.add_column("timestamp", style="magenta")
        for uid in uids:
            table.add_row(
                str(uid), 
                str( round(step_log['uid_data'][ str(uid) ]['average_loss'], 4)), 
                str( round(step_log['uid_data'][ str(uid) ]['win_rate'], 4)),
                str(step_log['uid_data'][ str(uid) ]['win_total']),
                str( round(self.weights[uid].item(), 4) ),
                str( round(step_log['uid_data'][ str(uid) ]['last_update'], 0)),
                str( step_log['uid_data'][ str(uid) ]['timestamp']),
            )
        console = Console()
        console.print(table)

        ws, ui = self.weights.topk( len( self.weights ))
        table = Table(title="Weights > 0.001")
        table.add_column("uid", justify="right", style="cyan", no_wrap=True)
        table.add_column("weight", style="magenta")
        for index, weight in list(zip(ui.tolist(), ws.tolist())):
            if weight > 0.001:
                table.add_row(str(index), str(round(weight, 4)))
        console = Console()
        console.print(table)

        # Sink step log.
        bt.logging.trace(f"Step results: {step_log}")
        original_format_json = json.dumps(step_log)
        uids = step_log['uids']
        uid_data = step_log['uid_data']

        # Create a new dictionary with the required format
        graphed_data = {
            'time': time.time(),
            'block': self.subtensor.block,
            'uid_data': {str(uid): uid_data[str(uid)]['average_loss'] for uid in uids},
            'weight_data': {str(uid): self.weights[uid].item() for uid in uids}
        }

        if self.config.wandb.on: 
            bt.logging.trace('Logging to Wandb')
            self.wandb_run.log({ **graphed_data, "original_format_json": original_format_json}, step=self.global_step)
            bt.logging.trace('finished log to Wandb')
        bt.logging.debug('Finished run step.')

    async def run(self):
        while True:
            try:            
                while self.metagraph.block.item() - self.last_epoch < self.config.blocks_per_epoch:
                    await self.try_run_step( ttl = RUN_STEP_MAX_TIME )
                    await self.try_sync_metagraph( ttl = 60 )
                    bt.logging.debug(f"{self.metagraph.block.item() - self.last_epoch } / {self.config.blocks_per_epoch} blocks until next epoch.")
                    self.global_step += 1

                if not self.config.dont_set_weights:
                    await self.try_set_weights( ttl = 60 )
                self.last_epoch = self.metagraph.block.item()
                self.epoch_step += 1

            except KeyboardInterrupt:
                bt.logging.info("KeyboardInterrupt caught, gracefully closing the wandb run...")
                if self.config.wandb.on: self.wandb_run.finish()
                exit()

            except Exception as e:
                bt.logging.error(f"Error in validator loop \n {e} \n {traceback.format_exc()}")


if __name__ == "__main__":
    asyncio.run( Validator().run() ) 