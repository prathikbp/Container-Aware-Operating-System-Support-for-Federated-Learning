import docker
import time
import sys
import os

def get_project_name():
    """
    Tries to infer the docker-compose project name.
    Defaults to the directory name if 'COMPOSE_PROJECT_NAME' env var isn't set.
    """
    project_name = os.environ.get('COMPOSE_PROJECT_NAME')
    if project_name:
        return project_name
    
    # Fallback: get the current directory name
    try:
        return os.path.basename(os.getcwd())
    except Exception:
        # A final fallback in case getcwd() fails
        print("[Manager] Could not infer project name, defaulting to 'fl-project'")
        return "fl-project"

def find_client_containers(client, project_name):
    """Find all running containers for the 'client' service."""
    client_containers = []
    try:
        print(f"[Manager] Searching for containers with label 'com.docker.compose.project={project_name}' and 'com.docker.compose.service=client'...")
        containers = client.containers.list(
            filters={
                "label": [
                    # name of the project TODO: make dynamic or add in config
                    f"com.docker.compose.project=container-aware-operating-system-support-for-federated-learning",
                    "com.docker.compose.service=client"
                ],
                "status": "running"
            }
        )
        return containers
    except docker.errors.NotFound:
        print("[Manager] Could not find any containers. Is Docker running?")
        return []
    except Exception as e:
        print(f"[Manager] Error listing containers: {e}")
        return []

def throttle_client(container, cpu_quota=10000):
    """Throttle a specific container to a given CPU quota."""
    try:
        print(f"[Manager] Throttling container {container.name} to CPU quota: {cpu_quota}...")
        # cpu_quota=10000 means 10000/100000 = 0.1 = 10% CPU
        container.update(cpu_quota=cpu_quota, cpu_period=100000)
        print(f"[Manager] Successfully throttled {container.name}.")
    except Exception as e:
        print(f"[Manager] Error throttling container {container.name}: {e}")

def main():
    print("[Manager] Starting Straggler Manager...")
    
    # The 'host.docker.internal' DNS name is a special name that
    # containers can use to connect to the host's Docker daemon.
    # We must mount the Docker socket in docker-compose.yml for this to work.
    try:
        client = docker.from_env()
        client.ping()
        print("[Manager] Successfully connected to Docker daemon.")
    except Exception as e:
        print(f"[Manager] Failed to connect to Docker daemon.")
        print(f"[Manager] Error: {e}")
        print("[Manager] Please ensure the Docker socket is mounted: - /var/run/docker.sock:/var/run/docker.sock")
        sys.exit(1)

    project_name = get_project_name()
    print(f"[Manager] Inferred project name: {project_name}")
    
    # Wait for client containers to start
    client_containers = []
    max_retries = 10
    retry_count = 0
    
    while not client_containers and retry_count < max_retries:
        print(f"[Manager] Waiting for client containers to start (Attempt {retry_count+1}/{max_retries})...")
        client_containers = find_client_containers(client, project_name)
        if not client_containers:
            time.sleep(5) # Wait 5 seconds before retrying
            retry_count += 1
            
    if not client_containers:
        print("[Manager] Could not find any client containers after waiting. Exiting.")
        sys.exit(1)

    print(f"[Manager] Found {len(client_containers)} client containers:")
    for c in client_containers:
        print(f"  - {c.name}")
    
    # --- The Experiment ---
    # We will throttle the first client in the list to be our "straggler"
    straggler_container = client_containers[0]
    cpu_quota_50_percent = 50000

    throttle_client(straggler_container, cpu_quota_50_percent)

    print("\n" + "="*50)
    print(f"[Manager] Experiment is running.")
    print(f"[Manager] Straggler: {straggler_container.name} is limited to 50% CPU.")
    print(f"[Manager] Other {len(client_containers) - 1} clients are running normally.")
    print("[Manager] This manager will now exit, but the throttle will remain.")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()