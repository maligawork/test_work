def calculate_accumulate_steps(batch_size: int,
                               weight_decay: float = 5e-4,
                               nominal_batch_size: int = 64,
                               ):
    """Calculates a number of accumulated steps and corrects weight decay for the gradient accumulation feature.
        Args:
            batch_size (int): batch size.
            weight_decay (float): weight decay coefficient.
            nominal_batch_size (int): nominal batch size for the gradient accumulation.
        Returns:
            num_accumulate_steps (int): number of accumulated steps for the gradient accumulation.
            scaled_weight_decay (float): corrected weight decay coefficient.
    """

    if nominal_batch_size <= batch_size:
        return 1, weight_decay

    num_accumulate_steps = max(round(nominal_batch_size / batch_size), 1)
    scaled_weight_decay = weight_decay * batch_size * num_accumulate_steps / nominal_batch_size

    return num_accumulate_steps, scaled_weight_decay
