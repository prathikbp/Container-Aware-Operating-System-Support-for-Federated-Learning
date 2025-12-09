#!/bin/bash

# This script maps container IDs in Prometheus config to readable names


# Get current containers
containers=$(docker ps --format "{{.ID}} {{.Names}}")

# Create base prometheus.yml
cat > prometheus.yml << 'EOF'
global:
  scrape_interval: 10s

scrape_configs:
  # FL CLIENTS
  - job_name: 'fl-clients'
    docker_sd_configs:
      - host: unix:///var/run/docker.sock
    relabel_configs:
      - source_labels: [__meta_docker_container_label_com_docker_compose_service]
        action: keep
        regex: client

      - source_labels: [__address__]
        action: replace
        target_label: __address__
        regex: ([^:]+):.*
        replacement: $1:8000

      - source_labels: [__meta_docker_container_name]
        action: replace
        target_label: instance
        regex: '.+client-([0-9]+)'
        replacement: 'client-$1'

  # FL SERVER
  - job_name: 'fl-server'
    static_configs:
      - targets: 
        - 'server:8081'
        labels:
          instance: 'fl-server'

  # CADVISOR WITH CONTAINER ID MAPPING
  - job_name: 'cadvisor'
    static_configs:
      - targets:
        - 'cadvisor:8080'
    metric_relabel_configs:
EOF

# Add container mappings
echo "$containers" | while read container_id container_name; do
    if [ -z "$container_id" ]; then
        continue
    fi
    
    # Get short ID (12 chars)
    short_id=$(echo "$container_id" | cut -c1-12)
    
    # Determine readable name
    case "$container_name" in
        *server*) readable_name="fl-server" ;;
        *client-1*) readable_name="client-1" ;;
        *client-2*) readable_name="client-2" ;;
        *client-3*) readable_name="client-3" ;;
        *client*) readable_name="client" ;;
        *grafana*) readable_name="grafana" ;;
        *prometheus*) readable_name="prometheus" ;;
        *cadvisor*) readable_name="cadvisor" ;;
        *redis*) readable_name="redis" ;;
        *manager*) readable_name="manager" ;;
        *) readable_name="$container_name" ;;
    esac
    
    # Add mapping rules to prometheus.yml
    cat >> prometheus.yml << EOF
      # Map container ID $short_id to $readable_name
      - source_labels: [id]
        action: replace
        target_label: container_name
        regex: '.*$short_id.*'
        replacement: '$readable_name'
      - source_labels: [id]
        action: replace
        target_label: id
        regex: '.*/docker/$short_id.*'
        replacement: '$readable_name'
EOF
    
    echo "Mapped: $short_id -> $readable_name"
done

echo "Prometheus configuration updated"

# Restart Prometheus if running
if docker ps --format "{{.Names}}" | grep -q prometheus; then
    docker-compose restart prometheus
    echo "Prometheus restarted"
else
    echo "Prometheus not running. Start with: docker-compose up -d prometheus"
fi

