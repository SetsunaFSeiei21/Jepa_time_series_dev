import json
import logging
import os
from datetime import datetime
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch

from .data import BatchData


class BaseEngine:
    """Base engine with common functionality."""

    def __init__(
        self,
        device: torch.device,
        model: torch.nn.Module,
        logger: logging.Logger,
        save_path: str,
    ):
        self.device = device
        self.model = model.to(device)
        self.logger = logger
        self.save_path = save_path

        # Create save directory
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        # Initialize JSON log file
        self.log_file = os.path.join(self.save_path, "training_log.json")

        # Store model parameter count if available
        if hasattr(model, "param_num"):
            self.logger.info(f"Model parameters: {model.param_num():,}")

    def initialize_log_file(self, log_metadata=None):
        """Initialize the JSON log file with metadata."""
        if os.path.exists(self.log_file):
            self.logger.info(
                f"Log file already exists at {self.log_file}, appending to it."
            )
        else:
            if log_metadata is None:
                log_metadata = {}
            with open(self.log_file, "w") as f:
                json.dump(log_metadata, f, indent=2)
            self.logger.info(f"Created new log file at {self.log_file}")

    def _log_epoch_info(self, epoch: Union[int, str], epoch_log: Dict):
        """
        Log epoch information to JSON file.

        Args:
            epoch: Current epoch number
            epoch_log: Information need to be recorded.
        """
        try:
            # Load existing log
            with open(self.log_file, "r") as f:
                log_data = json.load(f)

            # Append new log entry
            log_data["epoch_logs"][epoch] = epoch_log

            # Write back to file
            with open(self.log_file, "w") as f:
                json.dump(log_data, f, indent=2)

            self.logger.info(f"Logged epoch {epoch} info to {self.log_file}")

        except Exception as e:
            self.logger.error(f"Failed to log epoch info: {str(e)}")

    
