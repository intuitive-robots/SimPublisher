from __future__ import annotations

from ..core.utils import XRNodeRegistry
from .server import SimPubWebServer


def main() -> None:
    registry = XRNodeRegistry()
    server = SimPubWebServer(registry)
    server.serve_forever()


if __name__ == "__main__":
    main()
