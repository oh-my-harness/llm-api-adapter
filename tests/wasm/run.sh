#!/usr/bin/env bash
# Real-LLM wasm smoke test. Requires: wasm32-unknown-unknown target,
# wasm-bindgen-cli, node >= 20, and .env with OPENAI_API_KEY /
# OPENAI_BASE_URL / OPENAI_MODEL.
set -euo pipefail
cd "$(dirname "$0")"

# Load .env from repo root (same convention as test.sh).
set -a
[ -f ../../.env ] && source ../../.env
set +a

: "${OPENAI_API_KEY:?set OPENAI_API_KEY in .env}"
: "${OPENAI_BASE_URL:?set OPENAI_BASE_URL in .env}"
: "${OPENAI_MODEL:?set OPENAI_MODEL in .env}"

cargo build --target wasm32-unknown-unknown
wasm-bindgen target/wasm32-unknown-unknown/debug/wasm_smoke.wasm \
    --out-dir bindgen --target nodejs

node -e '
const m = require("./bindgen/wasm_smoke.js");
m.run_smoke(process.env.OPENAI_BASE_URL, process.env.OPENAI_API_KEY, process.env.OPENAI_MODEL)
  .then(report => {
    const ok = /^non-stream: WASM_OK/m.test(report) && /^stream: WASM_OK/m.test(report);
    console.log(ok ? "WASM SMOKE: PASS" : "WASM SMOKE: FAIL (expected WASM_OK)\n" + report);
    process.exit(ok ? 0 : 1);
  })
  .catch(e => { console.error("WASM SMOKE: FAIL:", e); process.exit(1); });
'
