#!/usr/bin/env bash
# Bypass macOS system HTTP proxy so wiremock's 127.0.0.1 server is reachable.
export NO_PROXY=127.0.0.1,localhost
exec cargo test "$@"
