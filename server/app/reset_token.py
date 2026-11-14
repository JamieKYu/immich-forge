"""Make a new Forge token, for when the old one is lost.

    docker exec immich-forge python -m app.reset_token

Prints the new token. The running server uses it straight away (it re-reads
the settings file), and the old token stops working — paste the new one into
the Forge extension and sign in to the setup page with it.
"""
from __future__ import annotations

import sys

from .config import get_settings
from .setup_config import SetupConfig


def main() -> int:
    setup = SetupConfig(get_settings())
    if setup.from_env("forge_api_token"):
        print(
            "The Forge token is set by the FORGE_API_TOKEN environment variable, "
            "so it can't be reset here. Change it in your docker run / compose "
            "settings and recreate the container.",
            file=sys.stderr,
        )
        return 1
    token = setup.regenerate_token()
    print("New Forge token (the old one no longer works):")
    print(token)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
