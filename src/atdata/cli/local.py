"""Local infrastructure management for atdata.

This module provides commands to start and stop local development infrastructure:
- Redis: For index storage and metadata
- MinIO: S3-compatible object storage for dataset files

The infrastructure runs in Docker containers managed via docker-compose or
direct docker commands.
"""

import shutil
import subprocess
import sys
from pathlib import Path
from textwrap import dedent

# Container names for tracking
REDIS_CONTAINER = "atdata-redis"
MINIO_CONTAINER = "atdata-minio"

# Docker compose configuration
COMPOSE_TEMPLATE = dedent("""\
    version: '3.8'

    services:
      redis:
        image: redis:7-alpine
        container_name: {redis_container}
        ports:
          - "{redis_port}:6379"
        volumes:
          - atdata-redis-data:/data
        command: redis-server --appendonly yes --maxmemory-policy noeviction
        restart: unless-stopped
        healthcheck:
          test: ["CMD", "redis-cli", "ping"]
          interval: 5s
          timeout: 3s
          retries: 3

      minio:
        image: minio/minio:latest
        container_name: {minio_container}
        ports:
          - "{minio_port}:9000"
          - "{minio_console_port}:9001"
        volumes:
          - atdata-minio-data:/data
        environment:
          MINIO_ROOT_USER: minioadmin
          MINIO_ROOT_PASSWORD: minioadmin
        command: server /data --console-address ":9001"
        restart: unless-stopped
        healthcheck:
          test: ["CMD", "curl", "-f", "http://localhost:9000/minio/health/live"]
          interval: 5s
          timeout: 3s
          retries: 3

    volumes:
      atdata-redis-data:
      atdata-minio-data:
""")


def _check_docker() -> bool:
    """Check if Docker is available and running."""
    if not shutil.which("docker"):
        print("Error: Docker is not installed or not in PATH", file=sys.stderr)
        return False

    try:
        result = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            print("Error: Docker daemon is not running", file=sys.stderr)
            return False
    except subprocess.TimeoutExpired:
        print("Error: Docker daemon not responding", file=sys.stderr)
        return False
    except Exception as e:
        print(f"Error checking Docker: {e}", file=sys.stderr)
        return False

    return True


def _get_compose_file(
    redis_port: int,
    minio_port: int,
    minio_console_port: int,
) -> str:
    """Generate docker-compose configuration."""
    return COMPOSE_TEMPLATE.format(
        redis_container=REDIS_CONTAINER,
        minio_container=MINIO_CONTAINER,
        redis_port=redis_port,
        minio_port=minio_port,
        minio_console_port=minio_console_port,
    )


def _container_running(name: str) -> bool:
    """Check if a container is running."""
    try:
        result = subprocess.run(
            ["docker", "inspect", "-f", "{{.State.Running}}", name],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.returncode == 0 and result.stdout.strip() == "true"
    except (OSError, subprocess.SubprocessError):
        return False


def _run_compose(
    compose_content: str,
    command: list[str],
    *,
    capture_output: bool = False,
) -> subprocess.CompletedProcess:
    """Run a docker-compose command with the given configuration."""
    # Write compose file to temp location
    compose_dir = Path.home() / ".atdata"
    compose_dir.mkdir(exist_ok=True)
    compose_file = compose_dir / "docker-compose.yml"
    compose_file.write_text(compose_content)

    # Prefer 'docker compose' (v2) over 'docker-compose' (v1)
    if shutil.which("docker"):
        # Check if docker compose v2 is available
        check = subprocess.run(
            ["docker", "compose", "version"],
            capture_output=True,
            timeout=5,
        )
        if check.returncode == 0:
            base_cmd = ["docker", "compose"]
        elif shutil.which("docker-compose"):
            base_cmd = ["docker-compose"]
        else:
            raise RuntimeError(
                "Neither 'docker compose' nor 'docker-compose' available"
            )
    else:
        raise RuntimeError("Docker not found")

    full_cmd = base_cmd + ["-f", str(compose_file)] + command

    return subprocess.run(
        full_cmd,
        capture_output=capture_output,
        text=True,
    )


def local_up(
    redis_port: int = 6379,
    minio_port: int = 9000,
    minio_console_port: int = 9001,
    detach: bool = True,
) -> int:
    """Start local development infrastructure.

    Args:
        redis_port: Port for Redis (default: 6379)
        minio_port: Port for MinIO API (default: 9000)
        minio_console_port: Port for MinIO console (default: 9001)
        detach: Run in background (default: True)

    Returns:
        Exit code (0 for success)
    """
    if not _check_docker():
        return 1

    print("Starting atdata local infrastructure...")

    compose_content = _get_compose_file(redis_port, minio_port, minio_console_port)
    command = ["up"]
    if detach:
        command.append("-d")

    try:
        result = _run_compose(compose_content, command)
        if result.returncode != 0:
            print("Error: Failed to start containers", file=sys.stderr)
            return result.returncode
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    # Wait a moment for containers to be healthy
    import time

    time.sleep(2)

    # Show status
    print()
    print("Local infrastructure started:")
    print(f"  Redis:        localhost:{redis_port}")
    print(f"  MinIO API:    http://localhost:{minio_port}")
    print(f"  MinIO Console: http://localhost:{minio_console_port}")
    print()
    print("MinIO credentials: minioadmin / minioadmin")
    print()
    print("Example usage:")
    print("  from atdata.local import Index, S3DataStore")
    print("  ")
    print("  store = S3DataStore.from_credentials({")
    print(f"      'AWS_ENDPOINT': 'http://localhost:{minio_port}',")
    print("      'AWS_ACCESS_KEY_ID': 'minioadmin',")
    print("      'AWS_SECRET_ACCESS_KEY': 'minioadmin',")
    print("  }, bucket='datasets')")
    print("  index = Index(data_store=store)")

    return 0


def local_down(remove_volumes: bool = False) -> int:
    """Stop local development infrastructure.

    Args:
        remove_volumes: Also remove data volumes (default: False)

    Returns:
        Exit code (0 for success)
    """
    if not _check_docker():
        return 1

    print("Stopping atdata local infrastructure...")

    # Use default ports for compose file (actual ports don't matter for down)
    compose_content = _get_compose_file(6379, 9000, 9001)
    command = ["down"]
    if remove_volumes:
        command.append("-v")
        print("Warning: This will delete all local data!")

    try:
        result = _run_compose(compose_content, command)
        if result.returncode != 0:
            print("Error: Failed to stop containers", file=sys.stderr)
            return result.returncode
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    print("Local infrastructure stopped.")
    return 0


def local_status() -> int:
    """Show status of local infrastructure.

    Returns:
        Exit code (0 for success)
    """
    if not _check_docker():
        return 1

    redis_running = _container_running(REDIS_CONTAINER)
    minio_running = _container_running(MINIO_CONTAINER)

    print("atdata local infrastructure status:")
    print()
    print(f"  Redis ({REDIS_CONTAINER}):  {'running' if redis_running else 'stopped'}")
    print(f"  MinIO ({MINIO_CONTAINER}):  {'running' if minio_running else 'stopped'}")

    if redis_running or minio_running:
        print()
        print("To stop: atdata local down")
    else:
        print()
        print("To start: atdata local up")

    return 0
