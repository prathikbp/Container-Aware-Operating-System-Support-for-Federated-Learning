import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Sequence

import docker


@dataclass
class ManagerConfig:
    """Holds all tunable knobs for the straggler manager."""

    project_name: str
    service_name: str = "client"
    max_retries: int = 10
    retry_interval: int = 5
    cpu_quota: int = 50000  # 50% of a single CPU because 50000/100000.
    cpu_period: int = 100000
    straggler_index: int = 0

    @staticmethod
    def infer_project_name() -> str:
        """Best effort attempt to infer docker-compose project name."""
        project_name = os.environ.get("COMPOSE_PROJECT_NAME")
        if project_name:
            return project_name

        try:
            return os.path.basename(os.getcwd())
        except Exception:
            print("[Manager] Could not infer project name, defaulting to 'fl-project'")
            return "fl-project"


class StragglerManager:
    """Encapsulates the orchestration logic so it can be reused or tested."""

    def __init__(self, config: ManagerConfig):
        self.config = config
        self.client = self._connect_to_docker()

    @staticmethod
    def _connect_to_docker():
        """Connect to Docker and exit immediately if it is unavailable."""
        try:
            client = docker.from_env()
            client.ping()
            print("[Manager] Successfully connected to Docker daemon.")
            return client
        except Exception as exc:
            print("[Manager] Failed to connect to Docker daemon.")
            print(f"[Manager] Error: {exc}")
            print("[Manager] Please ensure /var/run/docker.sock is mounted inside the container.")
            sys.exit(1)

    def _compose_filters(self) -> dict:
        return {
            "label": [
                f"com.docker.compose.project={self.config.project_name}",
                f"com.docker.compose.service={self.config.service_name}",
            ],
            "status": "running",
        }

    def _find_client_containers(self) -> List[docker.models.containers.Container]:
        """Fetch running containers for the configured service."""
        try:
            filters = self._compose_filters()
            print(f"[Manager] Searching for containers with filters: {filters}")
            return self.client.containers.list(filters=filters)
        except docker.errors.NotFound:
            print("[Manager] Could not find any containers. Is Docker running?")
        except Exception as exc:
            print(f"[Manager] Error listing containers: {exc}")
        return []

    def _wait_for_client_containers(self) -> Sequence[docker.models.containers.Container]:
        for attempt in range(1, self.config.max_retries + 1):
            print(f"[Manager] Waiting for client containers (Attempt {attempt}/{self.config.max_retries})...")
            containers = self._find_client_containers()
            if containers:
                return containers
            time.sleep(self.config.retry_interval)
        return []

    def _select_straggler(self, containers: Sequence[docker.models.containers.Container]):
        idx = min(self.config.straggler_index, len(containers) - 1)
        return containers[idx]

    def _throttle_container(self, container):
        try:
            print(
                f"[Manager] Throttling container {container.name} "
                f"to CPU quota {self.config.cpu_quota}/{self.config.cpu_period}..."
            )
            container.update(cpu_quota=self.config.cpu_quota, cpu_period=self.config.cpu_period)
            print(f"[Manager] Successfully throttled {container.name}.")
        except Exception as exc:
            print(f"[Manager] Error throttling container {container.name}: {exc}")

    def run(self):
        print("[Manager] Starting Straggler Manager...")
        print(f"[Manager] Inferred project name: {self.config.project_name}")
        containers = self._wait_for_client_containers()
        if not containers:
            print("[Manager] Could not find any client containers after waiting. Exiting.")
            sys.exit(1)

        print(f"[Manager] Found {len(containers)} client containers:")
        for container in containers:
            print(f"  - {container.name}")

        straggler = self._select_straggler(containers)
        self._throttle_container(straggler)
        self._print_summary(containers, straggler)

    def _print_summary(self, containers, straggler):
        print("\n" + "=" * 50)
        print("[Manager] Experiment is running.")
        print(
            f"[Manager] Straggler: {straggler.name} is limited to "
            f"{self.config.cpu_quota / self.config.cpu_period:.0%} CPU."
        )
        print(f"[Manager] Other {len(containers) - 1} clients are running normally.")
        print("[Manager] This manager will now exit, but the throttle will remain.")
        print("=" * 50 + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Manage straggler containers in a Flower experiment.")
    parser.add_argument("--config-file", default="fl_config.json", help="Path to the shared experiment config.")
    parser.add_argument("--project-name", help="Docker compose project name.")
    parser.add_argument("--service-name", help="Compose service whose containers should be managed.")
    parser.add_argument("--straggler-index", type=int, help="Index of the container to throttle.")
    parser.add_argument("--cpu-quota", type=int, help="Quota passed to docker update.")
    parser.add_argument("--cpu-period", type=int, help="CPU period for docker update.")
    parser.add_argument("--max-retries", type=int, help="How long to wait for containers to appear.")
    parser.add_argument("--retry-interval", type=int, help="Seconds to wait between retries.")
    return parser.parse_args()


def load_manager_settings(path: str) -> Dict[str, object]:
    """Load shared configuration and return the manager block."""
    try:
        with open(path, "r", encoding="utf-8") as config_file:
            config = json.load(config_file)
            manager_settings = config.get("manager", {})
            if not manager_settings:
                print(f"[Manager] Config file {path} missing 'manager' section; using defaults.")
            return manager_settings
    except FileNotFoundError:
        print(f"[Manager] Config file {path} not found.")
        sys.exit(1)
    except json.JSONDecodeError as exc:
        print(f"[Manager] Failed to parse config file {path}: {exc}")
        sys.exit(1)


def build_manager_config(args) -> ManagerConfig:
    settings = load_manager_settings(args.config_file)
    defaults = ManagerConfig(project_name="")

    project_name = args.project_name or settings.get("project_name") or ManagerConfig.infer_project_name()

    def coalesce(value, key):
        if value is not None:
            return value
        return settings.get(key, getattr(defaults, key))

    return ManagerConfig(
        project_name=project_name,
        service_name=coalesce(args.service_name, "service_name"),
        max_retries=coalesce(args.max_retries, "max_retries"),
        retry_interval=coalesce(args.retry_interval, "retry_interval"),
        cpu_quota=coalesce(args.cpu_quota, "cpu_quota"),
        cpu_period=coalesce(args.cpu_period, "cpu_period"),
        straggler_index=coalesce(args.straggler_index, "straggler_index"),
    )


def run_manager_cli():
    args = parse_args()
    config = build_manager_config(args)
    manager = StragglerManager(config)
    manager.run()
