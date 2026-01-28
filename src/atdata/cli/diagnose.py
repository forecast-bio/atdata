"""Diagnostic tools for atdata infrastructure.

This module provides commands to diagnose configuration issues with Redis
and other infrastructure components.
"""

import sys


def _print_status(label: str, ok: bool, detail: str = "") -> None:
    """Print a status line with checkmark or X."""
    symbol = "✓" if ok else "✗"
    status = f"{symbol} {label}"
    if detail:
        status += f": {detail}"
    print(status)


def diagnose_redis(host: str = "localhost", port: int = 6379) -> int:
    """Diagnose Redis configuration and connectivity.

    Checks for common issues that can cause data loss:
    - Connection issues
    - Persistence settings (AOF/RDB)
    - Memory eviction policy
    - Memory usage

    Args:
        host: Redis host (default: localhost)
        port: Redis port (default: 6379)

    Returns:
        Exit code (0 if all checks pass, 1 if any issues found)
    """
    print(f"Diagnosing Redis at {host}:{port}...")
    print()

    issues_found = False

    # Try to connect
    try:
        from redis import Redis

        redis = Redis(host=host, port=port, socket_connect_timeout=5)
        redis.ping()
        _print_status("Connection", True, "connected")
    except ImportError:
        print("Error: redis package not installed", file=sys.stderr)
        return 1
    except Exception as e:
        _print_status("Connection", False, str(e))
        print()
        print("Cannot connect to Redis. Make sure Redis is running:")
        print("  atdata local up")
        return 1

    # Check Redis version
    try:
        info = redis.info()
        version = info.get("redis_version", "unknown")
        _print_status("Version", True, version)
    except Exception as e:
        _print_status("Version", False, str(e))
        issues_found = True

    # Check persistence - AOF
    try:
        aof_enabled = redis.config_get("appendonly").get("appendonly", "no")
        aof_ok = aof_enabled == "yes"
        _print_status(
            "AOF Persistence",
            aof_ok,
            "enabled" if aof_ok else "DISABLED - data may be lost on restart!",
        )
        if not aof_ok:
            issues_found = True
    except Exception as e:
        _print_status("AOF Persistence", False, f"check failed: {e}")
        issues_found = True

    # Check persistence - RDB
    try:
        save_config = redis.config_get("save").get("save", "")
        rdb_ok = bool(save_config and save_config.strip())
        _print_status(
            "RDB Persistence",
            rdb_ok,
            f"configured ({save_config})" if rdb_ok else "DISABLED",
        )
        # RDB disabled is only a warning if AOF is enabled
    except Exception as e:
        _print_status("RDB Persistence", False, f"check failed: {e}")

    # Check memory policy
    try:
        policy = redis.config_get("maxmemory-policy").get("maxmemory-policy", "unknown")
        # Safe policies that won't evict index data
        safe_policies = {
            "noeviction",
            "volatile-lru",
            "volatile-lfu",
            "volatile-ttl",
            "volatile-random",
        }
        policy_ok = policy in safe_policies

        if policy_ok:
            _print_status("Memory Policy", True, policy)
        else:
            _print_status(
                "Memory Policy",
                False,
                f"{policy} - may evict index data! Use 'noeviction' or 'volatile-*'",
            )
            issues_found = True
    except Exception as e:
        _print_status("Memory Policy", False, f"check failed: {e}")
        issues_found = True

    # Check maxmemory setting
    try:
        maxmemory = redis.config_get("maxmemory").get("maxmemory", "0")
        maxmemory_bytes = int(maxmemory)
        if maxmemory_bytes == 0:
            _print_status("Max Memory", True, "unlimited")
        else:
            maxmemory_mb = maxmemory_bytes / (1024 * 1024)
            _print_status("Max Memory", True, f"{maxmemory_mb:.0f} MB")
    except Exception as e:
        _print_status("Max Memory", False, f"check failed: {e}")

    # Check current memory usage
    try:
        memory_info = redis.info("memory")
        used_memory = memory_info.get("used_memory_human", "unknown")
        peak_memory = memory_info.get("used_memory_peak_human", "unknown")
        _print_status("Memory Usage", True, f"{used_memory} (peak: {peak_memory})")
    except Exception as e:
        _print_status("Memory Usage", False, f"check failed: {e}")

    # Check number of atdata keys
    try:
        dataset_count = 0
        schema_count = 0
        for key in redis.scan_iter(match="LocalDatasetEntry:*", count=100):
            dataset_count += 1
        for key in redis.scan_iter(match="LocalSchema:*", count=100):
            schema_count += 1
        _print_status(
            "atdata Keys", True, f"{dataset_count} datasets, {schema_count} schemas"
        )
    except Exception as e:
        _print_status("atdata Keys", False, f"check failed: {e}")

    print()

    if issues_found:
        print("Issues found! Recommended configuration:")
        print()
        print("  # In redis.conf or via CONFIG SET:")
        print("  appendonly yes")
        print("  maxmemory-policy noeviction")
        print()
        print("  # Or use atdata's preconfigured local setup:")
        print("  atdata local up")
        return 1
    else:
        print("All checks passed. Redis is properly configured for atdata.")
        return 0
