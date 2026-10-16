from torch.utils.data import Subset


def _unwrap_dataset(dataset):
    while isinstance(dataset, Subset):
        dataset = dataset.dataset

    return dataset


def initialize_reference_spectrum_bank(
    model,
    train_loader,
    device,
):
    """
    Initialize all fixed reference spectra C.

    Ordinary models do not implement
    set_reference_series_bank(), so they are
    left unchanged.
    """

    if not hasattr(
        model,
        "set_reference_series_bank",
    ):
        return None

    train_dataset = _unwrap_dataset(
        train_loader.dataset
    )

    if not hasattr(
        train_dataset,
        "get_reference_prefixes",
    ):
        raise AttributeError(
            "Training dataset does not implement "
            "get_reference_prefixes()."
        )

    (
        reference_sequences,
        reference_masks,
    ) = train_dataset.get_reference_prefixes()

    model.set_reference_series_bank(
        reference_sequences=[
            sequence.to(device)
            for sequence
            in reference_sequences
        ],
        reference_masks=[
            mask.to(device)
            for mask
            in reference_masks
        ],
    )

    return {
        "reference_ratios": list(
            train_dataset.reference_ratios
        ),
        "reference_lengths": list(
            train_dataset.reference_lengths
        ),
    }