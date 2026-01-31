"""Command-line interface for atdata.

This module provides CLI commands for managing local development infrastructure,
inspecting datasets, and diagnosing configuration issues.

Commands:
    atdata local up      Start Redis and MinIO containers for local development
    atdata local down    Stop local development containers
    atdata local status  Show status of local infrastructure
    atdata diagnose      Check Redis configuration and connectivity
    atdata inspect       Show dataset summary information
    atdata schema show   Display dataset schema
    atdata schema diff   Compare two dataset schemas
    atdata preview       Preview first N samples of a dataset
    atdata version       Show version information
"""

import sys

import typer

# ---------------------------------------------------------------------------
# App hierarchy
# ---------------------------------------------------------------------------

app = typer.Typer(
    name="atdata",
    help="A loose federation of distributed, typed datasets.",
    add_completion=False,
    no_args_is_help=True,
)

local_app = typer.Typer(
    name="local",
    help="Manage local development infrastructure.",
    no_args_is_help=True,
)
app.add_typer(local_app, name="local")

schema_app = typer.Typer(
    name="schema",
    help="Show or compare dataset schemas.",
    no_args_is_help=True,
)
app.add_typer(schema_app, name="schema")


# ---------------------------------------------------------------------------
# Top-level commands
# ---------------------------------------------------------------------------


@app.command()
def version() -> None:
    """Show version information."""
    try:
        from atdata import __version__

        ver = __version__
    except ImportError:
        from importlib.metadata import version as pkg_version

        ver = pkg_version("atdata")

    print(f"atdata {ver}")


@app.command()
def inspect(
    url: str = typer.Argument(help="Dataset URL, local path, or atmosphere URI"),
) -> None:
    """Show dataset summary (sample count, schema, shards)."""
    from .inspect import inspect_dataset

    code = inspect_dataset(url=url)
    raise typer.Exit(code=code)


@app.command()
def preview(
    url: str = typer.Argument(help="Dataset URL, local path, or atmosphere URI"),
    limit: int = typer.Option(5, help="Number of samples to preview."),
) -> None:
    """Preview first N samples of a dataset."""
    from .preview import preview_dataset

    code = preview_dataset(url=url, limit=limit)
    raise typer.Exit(code=code)


@app.command()
def diagnose(
    host: str = typer.Option("localhost", help="Redis host."),
    port: int = typer.Option(6379, help="Redis port."),
) -> None:
    """Diagnose Redis configuration and connectivity."""
    from .diagnose import diagnose_redis

    code = diagnose_redis(host=host, port=port)
    raise typer.Exit(code=code)


# ---------------------------------------------------------------------------
# local sub-commands
# ---------------------------------------------------------------------------


@local_app.command()
def up(
    redis_port: int = typer.Option(6379, help="Redis port."),
    minio_port: int = typer.Option(9000, help="MinIO API port."),
    minio_console_port: int = typer.Option(9001, help="MinIO console port."),
    detach: bool = typer.Option(
        True, "--detach", "-d", help="Run containers in detached mode."
    ),
) -> None:
    """Start Redis and MinIO containers."""
    from .local import local_up

    code = local_up(
        redis_port=redis_port,
        minio_port=minio_port,
        minio_console_port=minio_console_port,
        detach=detach,
    )
    raise typer.Exit(code=code)


@local_app.command()
def down(
    volumes: bool = typer.Option(
        False, "--volumes", "-v", help="Also remove volumes (deletes all data)."
    ),
) -> None:
    """Stop local development containers."""
    from .local import local_down

    code = local_down(remove_volumes=volumes)
    raise typer.Exit(code=code)


@local_app.command()
def status() -> None:
    """Show status of local infrastructure."""
    from .local import local_status

    code = local_status()
    raise typer.Exit(code=code)


# ---------------------------------------------------------------------------
# schema sub-commands
# ---------------------------------------------------------------------------


@schema_app.command("show")
def schema_show(
    dataset_ref: str = typer.Argument(
        help="Dataset URL, local path, or index reference."
    ),
) -> None:
    """Display dataset schema."""
    from .schema import schema_show as _schema_show

    code = _schema_show(dataset_ref=dataset_ref)
    raise typer.Exit(code=code)


@schema_app.command("diff")
def schema_diff(
    url_a: str = typer.Argument(help="First dataset URL."),
    url_b: str = typer.Argument(help="Second dataset URL."),
) -> None:
    """Compare two dataset schemas."""
    from .schema import schema_diff as _schema_diff

    code = _schema_diff(url_a=url_a, url_b=url_b)
    raise typer.Exit(code=code)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """Main entry point for the atdata CLI.

    Args:
        argv: Command-line arguments. If None, uses sys.argv[1:].

    Returns:
        Exit code (0 for success, non-zero for errors).
    """
    try:
        if argv is not None:
            app(args=argv, standalone_mode=False)
        else:
            app(standalone_mode=False)
        return 0
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else 0
    except Exception:
        return 1


if __name__ == "__main__":
    sys.exit(main())
