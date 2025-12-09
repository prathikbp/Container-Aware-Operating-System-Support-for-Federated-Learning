import time
import flwr as fl
from typing import List, Tuple, Dict, Optional
from flwr.common import Metrics, Scalar
import argparse
import warnings
import logging
import os
import logging.handlers
from absl import logging as absl_logging
from prometheus_client import start_http_server, Histogram, Gauge, Counter


# metrics definitions

# We want to see this go DOWN when we implement timeouts
ROUND_DURATION = Histogram(
    'server_round_duration_seconds',
    'Time taken for one full FL round (fit + eval)',
    buckets=[5.0, 10.0, 20.0, 30.0, 60.0, 120.0, float("inf")]
)

# We want to see this go UP when we drop stragglers
CLIENT_FAILURES = Counter(
    'server_client_failures_total',
    'Total number of clients that failed or timed out'
)

#  General Monitoring (System Health)
GLOBAL_ACCURACY = Gauge(
    'server_global_model_accuracy',
    'Accuracy of the global model after aggregation'
)

CONNECTED_CLIENTS = Gauge(
    'server_connected_clients',
    'Number of clients selected for the current round'
)

def suppress_warnings():
    # Suppress all warnings
    warnings.filterwarnings("ignore")

    # Suppress Flower warnings
    logging.getLogger("flwr").setLevel(logging.ERROR)

    # Suppress gRPC warnings
    os.environ["GRPC_VERBOSITY"] = "ERROR"
    os.environ["GRPC_TRACE"] = "none"
    logging.getLogger("grpc").setLevel(logging.ERROR)

    # Suppress absl logging
    absl_logging.set_verbosity(absl_logging.ERROR)
    absl_logging.get_absl_handler().setLevel(logging.ERROR)

    # Suppress fork warnings
    logging.getLogger("multiprocessing").setLevel(logging.ERROR)

    # Remove all handlers associated with the root logger object
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)


def parse_args():
    parser = argparse.ArgumentParser(description='Flower server for federated learning')
    parser.add_argument('--server-address', type=str, default='0.0.0.0:8080',
                      help='Server address (default: 0.0.0.0:8080)')
    parser.add_argument('--num-rounds', type=int, default=5,
                      help='Number of rounds of federated learning (default: 5)')
    parser.add_argument('--fraction-fit', type=float, default=1.0,
                      help='Fraction of clients used for training (default: 1.0)')
    parser.add_argument('--fraction-evaluate', type=float, default=1.0,
                      help='Fraction of clients used for evaluation (default: 1.0)')
    parser.add_argument('--min-fit-clients', type=int, default=2,
                      help='Minimum number of clients for training (default: 2)')
    parser.add_argument('--min-evaluate-clients', type=int, default=2,
                      help='Minimum number of clients for evaluation (default: 2)')
    parser.add_argument('--min-available-clients', type=int, default=2,
                      help='Minimum number of available clients needed (default: 2)')
    return parser.parse_args()


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """
    A more robust aggregation function for weighted averaging of metrics.
    
    This function handles:
    1.  Aggregating *all* numerical metrics reported by clients (not just 'accuracy').
    2.  Safely skipping metrics that are not numerical (e.g., strings).
    3.  Handling the case of empty metrics lists or zero total examples.
    """
    print(f"\n[Metrics] Aggregating metrics from {len(metrics)} clients")
    if not metrics:
        print("[Metrics] No metrics to aggregate")
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


class LoggingStrategy(fl.server.strategy.FedAvg):
    """FedAvg strategy with basic logging."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Keep track of when the round started
        self.round_start_time = 0.0

    def configure_fit(self, server_round, parameters, client_manager):
        print(f"\n[Round {server_round}] Starting new training round")
        
        #  METRIC: Start Round Timer 
        self.round_start_time = time.time()
        
        #  METRIC: Connected Clients 
        # We can approximate this by how many clients we sample
        # (Note: client_manager.num_available() is better if accessible, but sampling works)
        # We'll just track how many we *asked* to fit
        return super().configure_fit(server_round, parameters, client_manager)
    
    def aggregate_fit(self, server_round, results, failures):
        print(f"\n[Round {server_round}] Aggregating training results from {len(results)} clients")

        #  METRIC: Connected Clients (Update) 
        total_participants = len(results) + len(failures)
        CONNECTED_CLIENTS.set(total_participants)

        if failures:
            failure_count = len(failures)
            print(f"[Round {server_round}] {len(failures)} clients failed during training")
            CLIENT_FAILURES.inc(failure_count)
        return super().aggregate_fit(server_round, results, failures)

    def aggregate_evaluate(self, server_round, results, failures):
        print(f"\n[Round {server_round}] Aggregating evaluation results from {len(results)} clients")
        if failures:
            failure_count = len(failures)
            print(f"[Round {server_round}] {len(failures)} clients failed during evaluation")
            CLIENT_FAILURES.inc(failure_count)

        # Call parent to get aggregated metrics
        agg_result = super().aggregate_evaluate(server_round, results, failures)
        #  METRIC: Global Accuracy 
        # agg_result is a tuple: (loss, metrics)
        if agg_result and agg_result[1]:
            metrics = agg_result[1]
            if "accuracy" in metrics:
                acc = metrics["accuracy"]
                GLOBAL_ACCURACY.set(acc)
                print(f"[Metrics] Round {server_round} Global Accuracy: {acc*100:.2f}%")

        #  METRIC: Round Duration 
        # We assume a round ends after evaluation
        if self.round_start_time > 0:
            duration = time.time() - self.round_start_time
            ROUND_DURATION.observe(duration)
            print(f"[Metrics] Round {server_round} Duration: {duration:.2f}s")
            self.round_start_time = 0.0 # Reset

        return agg_result

    def configure_evaluate(self, server_round, parameters, client_manager):
        print(f"\n[Round {server_round}] Starting evaluation")
        return super().configure_evaluate(server_round, parameters, client_manager)

def create_strategy(args):
    """Create a Flower strategy with the given arguments."""
    return LoggingStrategy(
        fraction_fit=args.fraction_fit,
        fraction_evaluate=args.fraction_evaluate,
        min_fit_clients=args.min_fit_clients,
        min_evaluate_clients=args.min_evaluate_clients,
        min_available_clients=args.min_available_clients,
        evaluate_metrics_aggregation_fn=weighted_average,
        
    )

# Start server
if __name__ == "__main__":
    # Suppress warnings
    suppress_warnings()
    
    # Parse command line arguments
    args = parse_args()
    
    print("\n" + "="*50)
    print("Flower Server Configuration:")
    print(f"Server Address: {args.server_address}")
    print(f"Number of Rounds: {args.num_rounds}")
    print(f"Fraction Fit: {args.fraction_fit}")
    print(f"Fraction Evaluate: {args.fraction_evaluate}")
    print(f"Minimum Fit Clients: {args.min_fit_clients}")
    print(f"Minimum Evaluate Clients: {args.min_evaluate_clients}")
    print(f"Minimum Available Clients: {args.min_available_clients}")
    print("="*50 + "\n")

    #  METRIC: Start Server-Side Metrics 
    print("Starting Prometheus metrics server on port 8081...")
    start_http_server(8081)

    # Create strategy with parsed arguments
    strategy = create_strategy(args)

    print("Starting Flower server...")
    fl.server.start_server(
        server_address=args.server_address,
        config=fl.server.ServerConfig(num_rounds=args.num_rounds),
        strategy=strategy,
    )
