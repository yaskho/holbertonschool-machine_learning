#!/usr/bin/env python3

def early_stopping(cost, opt_cost, threshold, patience, count):
    """
    Determines if early stopping should occur

    cost: current validation cost
    opt_cost: lowest recorded validation cost
    threshold: minimum improvement required
    patience: max allowed non-improvement steps
    count: current number of consecutive non-improving steps

    Returns: (stop_early, updated_count)
    """

    # Check if improvement is significant
    if opt_cost - cost > threshold:
        # Improvement → reset counter
        return (False, 0)

    # No sufficient improvement → increase counter
    count += 1

    # Stop if patience exceeded
    if count >= patience:
        return (True, patience)

    return (False, count)
