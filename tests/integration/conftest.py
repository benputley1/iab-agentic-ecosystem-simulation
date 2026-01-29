"""
Integration test configuration.

This conftest overrides the parent's Redis skip behavior for tests
that use mock Redis instead of requiring real Redis.
"""

import pytest


def pytest_collection_modifyitems(config, items):
    """
    Override the parent conftest's skip behavior.

    Tests in this directory that use mock Redis should not be skipped.
    Only tests explicitly marked with @pytest.mark.requires_redis should
    be skipped when Redis is unavailable.
    """
    import socket

    def redis_available():
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            result = sock.connect_ex(('localhost', 6379))
            sock.close()
            return result == 0
        except Exception:
            return False

    if redis_available():
        return

    skip_redis = pytest.mark.skip(reason="Redis not available")

    for item in items:
        # Only skip tests that explicitly require Redis
        if "requires_redis" in item.keywords:
            item.add_marker(skip_redis)
        # Remove any skip markers that were added by parent conftest
        # for tests that just happen to be in the integration folder
        elif hasattr(item, '_markers'):
            item._markers = [m for m in item._markers if 'Redis not available' not in str(m)]
