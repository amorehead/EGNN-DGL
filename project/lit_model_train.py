import os
from pathlib import Path

import pytorch_lightning as pl
import torch.autograd
import torch.nn as nn
from pytorch_lightning.plugins import DDPPlugin

from project.datasets.QM9.qm9_dgl_data_module import QM9DGLDataModule
from project.utils.modules import LitEGNN
from project.utils.utils import collect_args, process_args, construct_pl_logger


def main(args):
    # -----------
    # Data
    # -----------
    # Load QM9 data module
    qm9_data_module = QM9DGLDataModule(batch_size=args.batch_size, num_dataloader_workers=args.num_workers)
    qm9_data_module.setup()

    # ------------
    # Model
    # ------------
    # Assemble a dictionary of model arguments
    dict_args = vars(args)
    use_wandb_logger = args.logger_name.lower() == 'wandb'  # Determine whether the user requested to use WandB

    # Pick model and supply it with a dictionary of arguments
    model = LitEGNN(num_node_input_feats=qm9_data_module.num_node_features,
                    num_edge_input_feats=qm9_data_module.num_edge_features,
                    gnn_activ_fn=nn.SiLU(),
                    num_gnn_layers=dict_args['num_gnn_layers'],
                    num_gnn_hidden_channels=dict_args['num_gnn_hidden_channels'],
                    num_gnn_attention_heads=dict_args['num_gnn_attention_heads'],
                    num_epochs=dict_args['num_epochs'],
                    metric_to_track=args.metric_to_track,
                    weight_decay=dict_args['weight_decay'],
                    lr=dict_args['lr'])
    args.experiment_name = f'LitEGNN-b{args.batch_size}-gl{args.num_gnn_layers}' \
                           f'-n{args.num_gnn_hidden_channels}' \
                           f'-e{args.num_gnn_hidden_channels}' \
        if not args.experiment_name \
        else args.experiment_name
    litegnn_template_ckpt_filename_metric_to_track = f'{args.metric_to_track}:.3f'
    template_ckpt_filename = 'LitEGNN-{epoch:02d}-{' + litegnn_template_ckpt_filename_metric_to_track + '}'

    # ------------
    # Checkpoint
    # ------------
    ckpt_path = os.path.join(args.ckpt_dir, args.ckpt_name)
    ckpt_path_exists = os.path.exists(ckpt_path)
    ckpt_provided = args.ckpt_name != '' and ckpt_path_exists
    model = model.load_from_checkpoint(ckpt_path,
                                       use_wandb_logger=use_wandb_logger,
                                       batch_size=args.batch_size,
                                       lr=args.lr,
                                       weight_decay=args.weight_decay) if ckpt_provided else model

    # ------------
    # Trainer
    # ------------
    trainer = pl.Trainer.from_argparse_args(args)

    # -------------
    # Learning Rate
    # -------------
    if args.find_lr:
        lr_finder = trainer.tuner.lr_find(model, datamodule=qm9_data_module)  # Run learning rate finder
        fig = lr_finder.plot(suggest=True)  # Plot learning rates
        fig.savefig('optimal_lr.pdf')
        fig.show()
        model.hparams.lr = lr_finder.suggestion()  # Save optimal learning rate
        print(f'Optimal learning rate found: {model.hparams.lr}')

    # ------------
    # Logger
    # ------------
    pl_logger = construct_pl_logger(args)  # Log everything to an external logger
    trainer.logger = pl_logger  # Assign specified logger (e.g. TensorBoardLogger) to Trainer instance

    # -----------
    # Callbacks
    # -----------
    # Create and use callbacks
    mode = 'min' if 'mse' in args.metric_to_track else 'max'
    early_stop_callback = pl.callbacks.EarlyStopping(monitor=args.metric_to_track,
                                                     mode=mode,
                                                     min_delta=args.min_delta,
                                                     patience=args.patience)
    ckpt_callback = pl.callbacks.ModelCheckpoint(
        monitor=args.metric_to_track,
        mode=mode,
        verbose=True,
        save_last=True,
        save_top_k=3,
        filename=template_ckpt_filename  # Warning: May cause a race condition if calling trainer.test() with many GPUs
    )
    lr_monitor_callback = pl.callbacks.LearningRateMonitor(logging_interval='step', log_momentum=True)
    trainer.callbacks = [early_stop_callback, ckpt_callback, lr_monitor_callback]

    # ------------
    # Restore
    # ------------
    # If using WandB, download checkpoint artifact from their servers if the checkpoint is not already stored locally
    if use_wandb_logger and args.ckpt_name != '' and not os.path.exists(ckpt_path):
        checkpoint_reference = f'{args.entity}/{args.project_name}/model-{args.run_id}:best'
        artifact = trainer.logger.experiment.use_artifact(checkpoint_reference, type='model')
        artifact_dir = artifact.download()
        model = model.load_from_checkpoint(Path(artifact_dir) / 'model.ckpt',
                                           use_wandb_logger=use_wandb_logger,
                                           batch_size=args.batch_size,
                                           lr=args.lr,
                                           weight_decay=args.weight_decay)

    # -------------
    # Training
    # -------------
    # Train with the provided model and DataModule
    trainer.fit(model=model, datamodule=qm9_data_module)

    # -------------
    # Testing
    # -------------
    trainer.test()


if __name__ == '__main__':
    # -----------
    # Arguments
    # -----------
    # Collect all arguments
    parser = collect_args()

    # Parse all known and unknown arguments
    args, unparsed_argv = parser.parse_known_args()

    # Let the model add what it wants
    parser = LitEGNN.add_model_specific_args(parser)

    # Re-parse all known and unknown arguments after adding those that are model specific
    args, unparsed_argv = parser.parse_known_args()

    # Set Lightning-specific parameter values before constructing Trainer instance
    args.max_time = {'hours': args.max_hours, 'minutes': args.max_minutes}
    args.max_epochs = args.num_epochs
    args.profiler = args.profiler_method
    args.accelerator = args.multi_gpu_backend
    args.auto_select_gpus = args.auto_choose_gpus
    args.gpus = args.num_gpus
    args.num_nodes = args.num_compute_nodes
    args.precision = args.gpu_precision
    args.accumulate_grad_batches = args.accum_grad_batches
    args.gradient_clip_val = args.grad_clip_val
    args.gradient_clip_algo = args.grad_clip_algo
    args.stochastic_weight_avg = args.stc_weight_avg
    args.deterministic = True  # Make LightningModule's training reproducible

    # Set plugins for Lightning
    args.plugins = [
        # 'ddp_sharded',  # For sharded model training (to reduce GPU requirements)
        # We only select certain edge messages to update coordinates equivariantly, so we must ignore unused parameters
        DDPPlugin(find_unused_parameters=True)
    ]

    # Finalize all arguments as necessary
    args = process_args(args)

    # Begin execution of model training with given args
    main(args)
