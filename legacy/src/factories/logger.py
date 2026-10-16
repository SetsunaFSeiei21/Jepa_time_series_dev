import logging
import os
from dataclasses import dataclass
from typing import Optional, Dict, Any, Callable


@dataclass
class LoggerConfig:
    """Data class for logger configuration"""

    name: str
    log_file: str
    level: int = logging.INFO
    handlers: Optional[list] = None

    def __post_init__(self):
        if self.handlers is None:
            # self.handlers = ["console", "file"]
            self.handlers = ["file"]


class LoggerFactory:
    """Factory for creating loggers with flexible configuration"""

    # Default configurations for different modes
    MODE_CONFIGS = {
        "train": LoggerConfig(
            name="training_logger",
            log_file="training.log",
        ),
        "inference": LoggerConfig(
            name="inference_logger",
            log_file="inference.log",
        ),
        "pretrain": LoggerConfig(
            name="pretrain_logger",
            log_file="pretrain.log",
        ),
        "finetune": LoggerConfig(
            name="finetune_logger",
            log_file="finetune.log",
        ),
        "mae_pretrain": LoggerConfig(
            name="mae_pretrain_logger",
            log_file="mae_pretrain.log",
        ),
        "contrastive_pretrain": LoggerConfig(
            name="contrastive_pretrain_logger",
            log_file="contrastive_pretrain.log",
        ),
    }

    # Handler creators mapping
    _HANDLER_CREATORS = {
        "console": lambda config, **kwargs: logging.StreamHandler(),
        "file": lambda config, log_path, **kwargs: logging.FileHandler(log_path),
    }

    @classmethod
    def register_handler(cls, name: str, creator: Callable) -> None:
        """Register a custom handler creator"""
        cls._HANDLER_CREATORS[name] = creator

    @classmethod
    def create_logger(
        cls,
        output_dir: str,
        mode: str = "train",
        custom_config: Optional[Dict[str, Any]] = None,
    ) -> logging.Logger:
        """
        Build a logger

        Args:
            output_dir: Output directory for log files
            mode: Logger mode ('train', 'inference', or custom)
            custom_config: Custom configuration to override defaults

        Returns:
            Configured logger
        """
        # Get base configuration
        if mode in cls.MODE_CONFIGS:
            config = cls.MODE_CONFIGS[mode]
        else:
            # Use provided config or create default
            if custom_config:
                config = LoggerConfig(**custom_config)
            else:
                raise ValueError(f"Unknown mode '{mode}' and no custom config provided")

        # Merge with custom config if provided
        if custom_config:
            for key, value in custom_config.items():
                if hasattr(config, key):
                    setattr(config, key, value)

        # Create formatter
        formatter = logging.Formatter(
            fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # Create logger
        logger = logging.getLogger(config.name)
        logger.setLevel(config.level)

        # Clear existing handlers
        logger.handlers.clear()

        # Add configured handlers
        for handler_type in config.handlers:
            if handler_type not in cls._HANDLER_CREATORS:
                raise ValueError(f"Unknown handler type: {handler_type}")

            # Create handler
            if handler_type == "file":
                log_path = os.path.join(output_dir, config.log_file)
                os.makedirs(os.path.dirname(log_path), exist_ok=True)
                handler = cls._HANDLER_CREATORS[handler_type](
                    config=config, log_path=log_path
                )
            else:
                handler = cls._HANDLER_CREATORS[handler_type](config=config)

            # Configure handler
            handler.setLevel(config.level)
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger
