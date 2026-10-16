from hydra.utils import get_class, instantiate
from omegaconf import DictConfig, OmegaConf
from torch import Tensor

from src.graph import GraphSource, StaticGraph

from .base import BaseModel

GRAPH_KEY = "graph_transform"


def build_graph(model_cfg: DictConfig, adj: Tensor | None) -> GraphSource | None:
    """Graph source for models whose yaml sets `graph_transform`, otherwise None."""
    transform_cfg = model_cfg.get(GRAPH_KEY)
    if transform_cfg is None:
        return None
    if adj is None:
        raise ValueError(
            f"{model_cfg._target_} needs a graph but the dataset has none; "
            f"override model.{GRAPH_KEY}=null if the model can run without one."
        )
    return StaticGraph(adj, instantiate(transform_cfg))


def build_model(
    model_cfg: DictConfig,
    time_encoding_cfg: DictConfig,
    predictor_cfg: DictConfig,
    graph: GraphSource | None,
    *,
    num_nodes: int,
    in_channels: int,
    out_channels: int,
    time_feature_sizes: dict[str, int],
    his_len: int,
    pred_len: int,
) -> BaseModel:
    """Instantiates `model_cfg._target_` with its yaml hyperparameters plus the shared arguments.

    `time_feature_sizes` maps each time feature to its number of distinct values (meta.json).
    `predictor_cfg` holds the latent predictor widths shared by all models.
    """
    kwargs = OmegaConf.to_container(model_cfg, resolve=True)
    model_class = get_class(kwargs.pop("_target_"))
    kwargs.pop(GRAPH_KEY, None)
    if graph is not None:
        kwargs["num_supports"] = graph.num_supports
    model = model_class(
        **kwargs,
        num_nodes=num_nodes,
        in_channels=in_channels,
        out_channels=out_channels,
        num_time_features=len(time_feature_sizes),
        his_len=his_len,
        pred_len=pred_len,
        time_encoding=instantiate(time_encoding_cfg, feature_sizes=time_feature_sizes),
        predictor=OmegaConf.to_container(predictor_cfg, resolve=True),
    )
    assert isinstance(model, BaseModel), type(model)
    return model


__all__ = ["BaseModel", "build_graph", "build_model"]
