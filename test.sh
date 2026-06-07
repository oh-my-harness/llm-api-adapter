#!/usr/bin/env bash
# Bypass macOS system HTTP proxy so wiremock's 127.0.0.1 server is reachable.
export NO_PROXY=127.0.0.1,localhost
# Load .env so project-specific vars (e.g. ANTHROPIC_BASE_URL) override any
# conflicting values already present in the shell environment (e.g. from the
# Claude Code session which sets ANTHROPIC_BASE_URL to an internal gateway).
if [ -f .env ]; then
    set -a
    # shellcheck disable=SC1091
    source .env
    set +a
fi
exec cargo test "$@"
