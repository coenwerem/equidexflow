import argparse
from datetime import datetime
from omegaconf import OmegaConf
import os
import sys

from tensorboardX import SummaryWriter
import logging
import yaml
import random
import numpy as np
import torch

from equidexflow.loaders import get_dataloader
from equidexflow.models import get_model
from equidexflow.losses import get_losses
from equidexflow.utils.optimizers import get_optimizer
from equidexflow.metrics import get_metrics
from equidexflow.utils.logger import Logger
from equidexflow.trainers import get_trainer


def _main_dex(cfg, writer):
    """Training entry point for dexterous grasp generation (equidexflow_dex)."""
    from omegaconf import OmegaConf
    from equidexflow.models import get_dex_model
    from equidexflow.trainers.dex_trainer import DexGraspTrainer

    seed = cfg.get('seed', 1)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.set_num_threads(8)
    torch.backends.cudnn.deterministic = True

    # Build dataloaders (train / val / test)
    train_loader = get_dataloader('train', cfg.data.train)
    val_loader   = get_dataloader('val',   cfg.data.val)
    test_loader  = get_dataloader('test',  cfg.data.test) if 'test' in cfg.data else None

    # Build model
    model_cfg = cfg.model
    model = get_dex_model(
        p_uncond=float(model_cfg.get('p_uncond', 0.1)),
        guidance=float(model_cfg.get('guidance', 2.0)),
        num_ode_steps=int(model_cfg.get('num_ode_steps', 10)),
        hand_q_decoder_type=str(model_cfg.get('hand_q_decoder', 'deterministic')),
        n_coupling_layers=int(model_cfg.get('n_coupling_layers', 8)),
        surface_proj_tau=float(model_cfg.get('surface_proj_tau', 0.005)),
        wrist_frame=str(model_cfg.get('wrist_frame', 'base')),
    ).to(cfg.device)

    # Optionally resume from checkpoint
    if model_cfg.get('checkpoint', None) is not None:
        ckpt = torch.load(model_cfg.checkpoint, map_location='cpu', weights_only=False)
        state = ckpt.get('model', ckpt.get('model_state', ckpt))

        # Filter out keys whose shape no longer matches the current model.
        # Needed after architectural changes (e.g. wrist conditioning on
        # hand_q / contact decoder) when warm-starting from an old checkpoint:
        # strict=False allows missing keys but still raises on shape mismatch.
        current = model.state_dict()
        dropped = []
        clean_state = {}
        for k, v in state.items():
            if k in current and current[k].shape != v.shape:
                dropped.append((k, tuple(v.shape), tuple(current[k].shape)))
            else:
                clean_state[k] = v
        if dropped:
            logging.info("Dropping shape-mismatched checkpoint keys (will train from fresh init):")
            for k, old, new in dropped:
                logging.info(f"  {k}: ckpt {old}  ->  model {new}")

        missing, unexpected = model.load_state_dict(clean_state, strict=False)
        if missing:
            logging.info(f"Checkpoint missing keys (expected for new buffers / fresh decoders): {missing}")
        if unexpected:
            logging.warning(f"Checkpoint unexpected keys: {unexpected}")

    # Build optimizer
    tr = cfg.training
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(tr.lr),
        weight_decay=float(tr.weight_decay),
    )

    logger = Logger(writer)
    trainer = DexGraspTrainer(cfg, model, optimizer, train_loader, val_loader, test_loader, logger)
    trainer.run()


def main(cfg, writer):
    # Route dexterous training to the dedicated entry point
    model_name = cfg.model.get('name', '') if hasattr(cfg.model, 'get') else str(cfg.model)
    if model_name == 'equidexflow_dex':
        _main_dex(cfg, writer)
        return

    # ## Original ACRONYM / equidexflow training path (unchanged) ########
    # Setup seed
    seed = cfg.get('seed', 1)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.set_num_threads(8)
    torch.backends.cudnn.deterministic = True

    # Setup dataloader
    dataloaders = {}

    for split in ['train', 'val']:
        dataloaders[split] = get_dataloader(split, cfg.data[split])

    # Setup model
    model = get_model(cfg.model).to(cfg.device)

    # Setup losses
    losses = get_losses(cfg.losses)

    # Setup optimizer
    optimizer = get_optimizer(cfg.optimizer, model.parameters())

    # Setup metrics
    metrics = get_metrics(cfg.metrics)

    # Setup logger
    logger = Logger(writer)

    # Setup trainer
    trainer = get_trainer(cfg.trainer, cfg.device, dataloaders, model, losses, optimizer, metrics, logger)

    # Start learning
    trainer.run()


if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', type=str)
    parser.add_argument('--device', default=0)
    parser.add_argument('--logdir', default='outputs/training_results')
    parser.add_argument('--run', type=str, default=datetime.now().strftime('%Y%m%d-%H%M'))

    args = parser.parse_args()

    # Load and print config
    cfg = OmegaConf.load(args.config)
    print(OmegaConf.to_yaml(cfg))

    # Setup device
    if args.device == 'cpu':
        cfg.device = 'cpu'
    else:
        cfg.device = f'cuda:{args.device}'

    # Setup logdir
    config_filename = os.path.basename(args.config)
    config_basename = os.path.splitext(config_filename)[0]

    logdir = os.path.join(args.logdir, config_basename, args.run)

    # Setup tensorboard writer
    writer = SummaryWriter(logdir)

    # Setup logging
    logging.basicConfig(
        filename=os.path.join(logdir, 'logging.log'),
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y/%m/%d %I:%M:%S %p',
        level=logging.DEBUG
    )

    # Print logdir
    print(f"Result directory: {logdir}")
    logging.info(f"Result directory: {logdir}")

    # Save config
    config_path = os.path.join(logdir, config_filename)
    yaml.dump(yaml.safe_load(OmegaConf.to_yaml(cfg)), open(config_path, 'w'))

    print(f"Config saved as {config_path}")
    logging.info(f"Config saved as {config_path}")

    main(cfg, writer)
