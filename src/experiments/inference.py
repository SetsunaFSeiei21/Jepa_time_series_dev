import argparse
import os

import numpy as np
import torch
import yaml
from omegaconf import DictConfig, OmegaConf

from src.base.model import BaseModel
from src.engines.expert import ExpertEngine
from src.factories.loader import get_loaders
from src.factories.logger import LoggerFactory
from src.factories.model import ModelFactory
from src.utils.reference_spectrum import (
    initialize_reference_spectrum_bank,
)


def set_seed(seed: int) -> None:
    """
    Set random seeds for reproducible inference.
    """

    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(log_folder: str) -> DictConfig:
    """
    Load the Hydra configuration saved in the finetuning
    output directory.

    Expected path:

        log_folder/.hydra/config.yaml
    """

    config_path = os.path.join(
        log_folder,
        ".hydra",
        "config.yaml",
    )

    if not os.path.exists(config_path):
        raise FileNotFoundError(
            "Hydra configuration was not found: "
            f"{config_path}"
        )

    with open(
        config_path,
        "r",
        encoding="utf-8",
    ) as file:
        config_dict = yaml.safe_load(file)

    return OmegaConf.create(
        config_dict
    )


def load_model_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str,
    device,
    logger=None,
) -> torch.nn.Module:
    """
    Load only model parameters from a finetuning checkpoint.

    Supported formats:

        1. {
               "model_state_dict": ...
           }

        2. raw state_dict
    """

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            "Inference checkpoint was not found: "
            f"{checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    if (
        isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
    ):
        state_dict = checkpoint[
            "model_state_dict"
        ]

        checkpoint_epoch = checkpoint.get(
            "epoch",
            None,
        )

        best_val_loss = checkpoint.get(
            "best_val_loss",
            None,
        )

    else:
        state_dict = checkpoint
        checkpoint_epoch = None
        best_val_loss = None

    # The checkpoint and model architecture should match exactly.
    #
    # reference_spectra and reference_ready are non-persistent
    # buffers, so they do not participate in state_dict loading.
    model.load_state_dict(
        state_dict,
        strict=True,
    )

    if logger is not None:
        logger.info(
            "Loaded inference checkpoint: "
            f"{checkpoint_path}"
        )

        if checkpoint_epoch is not None:
            logger.info(
                "Checkpoint epoch: "
                f"{checkpoint_epoch}"
            )

        if best_val_loss is not None:
            logger.info(
                "Checkpoint best validation loss: "
                f"{best_val_loss}"
            )

    return model


def main(
    log_folder: str,
    checkpoint_path: str = "",
):
    """
    Run independent inference from a finetuning output
    directory.

    Required order:

        1. Load saved finetuning configuration.
        2. Rebuild datasets and dataloaders.
        3. Create the model.
        4. Load best finetuning weights.
        5. Rebuild reference spectrum bank C.
        6. Evaluate the test split.
    """

    # ------------------------------------------------------------
    # 1. Load saved finetuning configuration
    # ------------------------------------------------------------

    cfg = load_config(
        log_folder
    )

    dataset_kwargs = cfg["dataset"]
    model_kwargs = cfg["model"]
    exp_kwargs = cfg["exp"]
    task_exp = cfg["task"]

    seed = int(
        cfg.get(
            "seed",
            0,
        )
    )

    set_seed(seed)

    # Current task configuration uses:
    #
    #     his_len
    #     pred_len
    #
    # rather than the old seq_len/resolution/label interface.
    his_len = int(
        task_exp["his_len"]
    )

    pred_len = int(
        task_exp["pred_len"]
    )

    device_id = int(
        exp_kwargs.get(
            "device_id",
            0,
        )
    )

    device = (
        f"cuda:{device_id}"
        if torch.cuda.is_available()
        else "cpu"
    )

    meta_path = cfg["meta_path"]

    # ------------------------------------------------------------
    # 2. Create logger
    # ------------------------------------------------------------

    logger = LoggerFactory.create_logger(
        log_folder,
        mode="inference",
    )

    logger.info(
        "Starting independent inference."
    )

    logger.info(
        f"Device: {device}"
    )

    logger.info(
        f"Seed: {seed}"
    )

    # ------------------------------------------------------------
    # 3. Rebuild datasets and dataloaders
    # ------------------------------------------------------------
    #
    # Because this configuration comes from the finetuning
    # directory, exp_kwargs contains:
    #
    #     train_mode: finetune
    #     use_reference_bank: true
    #     checkpoint_ratios: [1.0]
    #
    # Therefore, the train dataset will expose the prefixes
    # needed to reconstruct C, and train/val/test samples will
    # carry the correct reference_index.
    # ------------------------------------------------------------

    (
        loaders,
        adj_mtx,
        norm_pipeline,
        num_features,
    ) = get_loaders(
        meta_path,
        dataset_kwargs,
        exp_kwargs,
        his_len,
        pred_len,
        device,
    )

    num_nodes = int(
        adj_mtx.shape[0]
    )

    # ------------------------------------------------------------
    # 4. Build model
    # ------------------------------------------------------------
    #
    # Keep this identical to finetune.py so that the checkpoint
    # architecture matches exactly.
    # ------------------------------------------------------------

    ext_kwargs = dict(
        model_kwargs,
        **task_exp,
        **{
            "node_num": num_nodes,
            "input_dim": (
                num_features + 1
            ),
            "output_dim": 1,
        },
    )

    model_name = model_kwargs[
        "_target_"
    ].split(".")[-1]

    model: BaseModel = (
        ModelFactory.create_model(
            model_name,
            ext_kwargs,
            adj_mtx,
            device,
        )
    )

    model = model.to(device)

    # ------------------------------------------------------------
    # 5. Resolve inference checkpoint
    # ------------------------------------------------------------

    if (
        checkpoint_path is None
        or checkpoint_path == ""
    ):
        checkpoint_path = os.path.join(
            log_folder,
            "best_model_checkpoint.pth",
        )

    # ------------------------------------------------------------
    # 6. Load validation-best finetuning weights
    # ------------------------------------------------------------

    model = load_model_checkpoint(
        model=model,
        checkpoint_path=checkpoint_path,
        device=device,
        logger=logger,
    )

    # ------------------------------------------------------------
    # 7. Rebuild C bank
    # ------------------------------------------------------------
    #
    # This must happen after checkpoint loading.
    #
    # For:
    #
    #     checkpoint_ratios: [1.0]
    #
    # this constructs:
    #
    #     C_1.0 = FFT(train_data[:T_train])
    #
    # Ordinary models do not implement set_reference_series_bank(),
    # so the helper returns None and leaves them unchanged.
    # ------------------------------------------------------------

    reference_info = (
        initialize_reference_spectrum_bank(
            model=model,
            train_loader=(
                loaders["train_loader"]
            ),
            device=device,
        )
    )

    if reference_info is not None:
        logger.info(
            "Rebuilt inference reference bank: "
            f"ratios="
            f"{reference_info['reference_ratios']}, "
            f"lengths="
            f"{reference_info['reference_lengths']}."
        )

        logger.info(
            "Reference bank tensor shape: "
            f"{tuple(model.reference_spectra.shape)}"
        )

        if not bool(
            model.reference_ready.item()
        ):
            raise RuntimeError(
                "Reference bank initialization "
                "did not mark the model as ready."
            )

    # ------------------------------------------------------------
    # 8. Build evaluation engine
    # ------------------------------------------------------------
    #
    # ExpertEngine.test() already contains the current masking,
    # normalization, inverse-normalization and metric logic.
    # ------------------------------------------------------------

    loss_func = (
        lambda x, y:
        torch.nn.functional.l1_loss(
            x,
            y,
            reduction="mean",
        )
    )

    engine = ExpertEngine(
        device=device,
        model=model,
        logger=logger,
        save_path=log_folder,
        scalar=norm_pipeline,
        train_loader=(
            loaders["train_loader"]
        ),
        val_loader=(
            loaders["val_loader"]
        ),
        test_loader=(
            loaders["test_loader"]
        ),
        loss_func=loss_func,
        config=exp_kwargs,
    )

    # ------------------------------------------------------------
    # 9. Test only
    # ------------------------------------------------------------

    logger.info(
        "Evaluating the validation-best model "
        "on the test split."
    )

    test_metrics = engine.test()

    logger.info(
        f"Inference test metrics: {test_metrics}"
    )

    print(
        "Inference test metrics:",
        test_metrics,
    )

    return test_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Run independent inference from a "
            "finetuning output directory."
        )
    )

    parser.add_argument(
        "--log_folder",
        type=str,
        required=True,
        help=(
            "Finetuning output directory containing "
            ".hydra/config.yaml and "
            "best_model_checkpoint.pth."
        ),
    )

    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="",
        help=(
            "Optional explicit checkpoint path. "
            "Defaults to "
            "<log_folder>/best_model_checkpoint.pth."
        ),
    )

    args = parser.parse_args()

    log_folder = os.path.abspath(
        os.path.expanduser(
            args.log_folder
        )
    )

    checkpoint_path = args.checkpoint_path

    if checkpoint_path:
        checkpoint_path = os.path.abspath(
            os.path.expanduser(
                checkpoint_path
            )
        )

    if not os.path.isdir(log_folder):
        raise NotADirectoryError(
            "The specified finetuning directory "
            f"does not exist: {log_folder}"
        )

    main(
        log_folder=log_folder,
        checkpoint_path=checkpoint_path,
    )