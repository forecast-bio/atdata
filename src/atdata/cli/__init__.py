"""Command-line interface for atdata.

This module provides CLI commands for managing local development infrastructure
and diagnosing configuration issues.

Commands:
    atdata local up     Start Redis and MinIO containers for local development
    atdata local down   Stop local development containers
    atdata diagnose     Check Redis configuration and connectivity
    atdata version      Show version information

Example:
    $ atdata local up
    Starting Redis on port 6379...
    Starting MinIO on port 9000...
    Local infrastructure ready.

    $ atdata diagnose
    Checking Redis configuration...
    ✓ Redis connected
    ✓ Persistence enabled (AOF)
    ✓ Memory policy: noeviction
"""

import argparse
import sys
from typing import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    """Main entry point for the atdata CLI.

    Args:
        argv: Command-line arguments. If None, uses sys.argv[1:].

    Returns:
        Exit code (0 for success, non-zero for errors).
    """
    parser = argparse.ArgumentParser(
        prog="atdata",
        description="A loose federation of distributed, typed datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--version", "-v",
        action="store_true",
        help="Show version information",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # 'local' command group
    local_parser = subparsers.add_parser(
        "local",
        help="Manage local development infrastructure",
    )
    local_subparsers = local_parser.add_subparsers(
        dest="local_command",
        help="Local infrastructure commands",
    )

    # 'local up' command
    up_parser = local_subparsers.add_parser(
        "up",
        help="Start Redis and MinIO containers",
    )
    up_parser.add_argument(
        "--redis-port",
        type=int,
        default=6379,
        help="Redis port (default: 6379)",
    )
    up_parser.add_argument(
        "--minio-port",
        type=int,
        default=9000,
        help="MinIO API port (default: 9000)",
    )
    up_parser.add_argument(
        "--minio-console-port",
        type=int,
        default=9001,
        help="MinIO console port (default: 9001)",
    )
    up_parser.add_argument(
        "--detach", "-d",
        action="store_true",
        default=True,
        help="Run containers in detached mode (default: True)",
    )

    # 'local down' command
    down_parser = local_subparsers.add_parser(
        "down",
        help="Stop local development containers",
    )
    down_parser.add_argument(
        "--volumes", "-v",
        action="store_true",
        help="Also remove volumes (deletes all data)",
    )

    # 'local status' command
    local_subparsers.add_parser(
        "status",
        help="Show status of local infrastructure",
    )

    # 'diagnose' command
    diagnose_parser = subparsers.add_parser(
        "diagnose",
        help="Diagnose Redis configuration and connectivity",
    )
    diagnose_parser.add_argument(
        "--host",
        default="localhost",
        help="Redis host (default: localhost)",
    )
    diagnose_parser.add_argument(
        "--port",
        type=int,
        default=6379,
        help="Redis port (default: 6379)",
    )

    # 'version' command (alternative to --version flag)
    subparsers.add_parser(
        "version",
        help="Show version information",
    )

    args = parser.parse_args(argv)

    # Handle --version flag
    if args.version or args.command == "version":
        return _cmd_version()

    # Handle 'local' commands
    if args.command == "local":
        if args.local_command == "up":
            return _cmd_local_up(
                redis_port=args.redis_port,
                minio_port=args.minio_port,
                minio_console_port=args.minio_console_port,
                detach=args.detach,
            )
        elif args.local_command == "down":
            return _cmd_local_down(remove_volumes=args.volumes)
        elif args.local_command == "status":
            return _cmd_local_status()
        else:
            local_parser.print_help()
            return 1

    # Handle 'diagnose' command
    if args.command == "diagnose":
        return _cmd_diagnose(host=args.host, port=args.port)

    # No command given
    parser.print_help()
    return 0


def _cmd_version() -> int:
    """Show version information."""
    try:
        from atdata import __version__
        version = __version__
    except ImportError:
        # Fallback to package metadata
        from importlib.metadata import version as pkg_version
        version = pkg_version("atdata")

    print(f"atdata {version}")
    return 0


def _cmd_local_up(
    redis_port: int,
    minio_port: int,
    minio_console_port: int,
    detach: bool,
) -> int:
    """Start local development infrastructure."""
    from .local import local_up
    return local_up(
        redis_port=redis_port,
        minio_port=minio_port,
        minio_console_port=minio_console_port,
        detach=detach,
    )


def _cmd_local_down(remove_volumes: bool) -> int:
    """Stop local development infrastructure."""
    from .local import local_down
    return local_down(remove_volumes=remove_volumes)


def _cmd_local_status() -> int:
    """Show status of local infrastructure."""
    from .local import local_status
    return local_status()


def _cmd_diagnose(host: str, port: int) -> int:
    """Diagnose Redis configuration."""
    from .diagnose import diagnose_redis
    return diagnose_redis(host=host, port=port)


if __name__ == "__main__":
    sys.exit(main())
