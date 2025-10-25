import flwr as fl
from typing import List, Tuple, Dict, Optional
from flwr.common import Metrics, Scalar


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """
    A more robust aggregation function for weighted averaging of metrics.
    
    This function handles:
    1.  Aggregating *all* numerical metrics reported by clients (not just 'accuracy').
    2.  Safely skipping metrics that are not numerical (e.g., strings).
    3.  Handling the case of empty metrics lists or zero total examples.
    """
    if not metrics:
        return {}

    # Initialize a dictionary to store the aggregated sums for each metric
    aggregated_sums: Dict[str, float] = {}
    total_examples: int = 0

    # Get all unique metric keys from the first client's metrics
    # We assume all clients report the same *types* of metrics,
    # but we will check for safety.
    all_keys = {key for _, m in metrics for key in m.keys()}

    # Initialize sums for all keys that are numeric
    for key in all_keys:
        # Check if the metric from the first client is a number
        is_numeric = False
        for _, m in metrics:
            if key in m and isinstance(m[key], (int, float)):
                is_numeric = True
                break  # Found a numeric instance, so we'll aggregate this key

        if is_numeric:
            aggregated_sums[key] = 0.0

    # Iterate through all client metrics
    for num_examples, client_metrics in metrics:
        total_examples += num_examples
        for key, value in client_metrics.items():
            # Only aggregate if it's a key we've identified as numeric
            if key in aggregated_sums and isinstance(value, (int, float)):
                aggregated_sums[key] += float(value) * num_examples

    # Avoid division by zero
    if total_examples == 0:
        return {key: 0.0 for key in aggregated_sums}

    # Calculate the weighted average for each metric
    final_metrics = {
        key: value / total_examples for key, value in aggregated_sums.items()
    }

    return final_metrics


# Define strategy
strategy = fl.server.strategy.FedAvg(
    fraction_fit=1.0,  # Sample 100% of available clients for training
    fraction_evaluate=1.0,  # Sample 100% of available clients for evaluation
    min_fit_clients=2,  # Minimum number of clients for training
    min_evaluate_clients=2,  # Minimum number of clients for evaluation
    min_available_clients=2,  # Minimum number of available clients needed to start round

    # Use the original function name with robust logic
    evaluate_metrics_aggregation_fn=weighted_average,
)

# Start server
if __name__ == "__main__":
    print("Starting Flower server...")
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=5),
        strategy=strategy,
    )
