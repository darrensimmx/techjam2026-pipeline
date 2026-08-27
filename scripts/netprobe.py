"""Control probe for the network block. Exit 0 only if every real network
operation is denied; exit 1 if any succeeds (which would make the block
vacuous, and any 'passes offline' claim resting on it worthless)."""
from __future__ import annotations

import socket
import sys
import urllib.request

ATTEMPTS = (
    ("TCP connect 1.1.1.1:443", lambda: socket.create_connection(("1.1.1.1", 443), timeout=5).close()),
    ("TCP connect 8.8.8.8:53", lambda: socket.create_connection(("8.8.8.8", 53), timeout=5).close()),
    ("DNS resolve pypi.org", lambda: socket.getaddrinfo("pypi.org", 443)),
    ("HTTPS GET example.com", lambda: urllib.request.urlopen("https://example.com", timeout=5).read(1)),
    ("UDP sendto 8.8.8.8:53", lambda: socket.socket(socket.AF_INET, socket.SOCK_DGRAM).sendto(b"\x00", ("8.8.8.8", 53))),
)


def main() -> int:
    reachable = []
    for name, attempt in ATTEMPTS:
        try:
            attempt()
            reachable.append(name)
            print(f"  {name:28} -> SUCCEEDED")
        except Exception as error:
            print(f"  {name:28} -> denied: {type(error).__name__}: {str(error)[:80]}")
    if reachable:
        print(f"NETWORK REACHABLE ({len(reachable)}/{len(ATTEMPTS)} succeeded) -- block is VACUOUS")
        return 1
    print(f"ALL {len(ATTEMPTS)} NETWORK OPERATIONS DENIED -- block is REAL")
    return 0


if __name__ == "__main__":
    sys.exit(main())
