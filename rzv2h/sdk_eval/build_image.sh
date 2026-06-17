#!/usr/bin/env bash
# Build the Renesas DRP-AI TVM (Mera2) Docker image for RZ/V2H.
#
# This is the *faithful* compile/runtime stack (real mera2 + MERA runtime).
# It needs two downloads that require a Renesas account login — put them in
# ./assets first (this script cannot download them for you). Both arrive as
# ZIPs and can be dropped in as-is:
#
#   DRP-AI Translator i8  (ZIP, contains DRP-AI_Translator_i8-*-Linux-x86_64-Install)
#     https://www.renesas.com/software-tool/drp-ai-translator-i8   (Downloads tab)
#   RZ/V2H AI SDK         (RTK0EF0180F*SJ.zip)
#     https://www.renesas.com/us/en/software-tool/rzv2h-ai-software-development-kit
#
# The repo Dockerfile COPYs every ./*.sh in the context and runs it, plus
# ./DRP-AI_Translator*-Install. So we assemble a CLEAN context holding only:
# Dockerfile + the SDK toolchain installer (.sh, from the AI SDK zip) + the
# Translator installer (from the Translator zip).
set -euo pipefail
cd "$(dirname "$0")"
ASSETS="${ASSETS:-./assets}"
CTX="${CTX:-./context}"
PRODUCT="${PRODUCT:-V2H}"
TAG="${TAG:-drpai-tvm-v2h}"

mkdir -p "$ASSETS"
TMPS=()
cleanup() { for d in "${TMPS[@]:-}"; do [[ -n "$d" ]] && rm -rf "$d"; done; }
trap cleanup EXIT

# --- DRP-AI Translator i8: accept an extracted *-Install or the downloaded zip ---
TR=$(ls "$ASSETS"/DRP-AI_Translator*-Linux*-x86_64-Install 2>/dev/null | head -n1 || true)
if [[ -z "$TR" ]]; then
  TRZIP=$(ls "$ASSETS"/*[Tt]ranslator*i8*.zip "$ASSETS"/*DRP-AI_Translator*.zip 2>/dev/null | head -n1 || true)
  if [[ -n "$TRZIP" ]]; then
    t=$(mktemp -d); TMPS+=("$t")
    unzip -o -q "$TRZIP" -d "$t"
    TR=$(find "$t" -iname "DRP-AI_Translator*-Linux*-x86_64-Install" | head -n1 || true)
  fi
fi

# --- RZ/V2H AI SDK zip (any v6.x build number) ---
ZIP=$(ls "$ASSETS"/RTK0EF0180F*SJ.zip 2>/dev/null | head -n1 || true)

if [[ -z "$TR" || -z "$ZIP" ]]; then
  echo "Missing gated downloads in $ASSETS (Renesas login required):" >&2
  [[ -z "$TR"  ]] && echo "  - DRP-AI Translator i8 (zip or extracted *-Install)" >&2
  [[ -z "$ZIP" ]] && echo "  - RZ/V2H AI SDK  (RTK0EF0180F*SJ.zip)" >&2
  exit 1
fi

# Clean build context.
rm -rf "$CTX" && mkdir -p "$CTX"
wget -nc https://raw.githubusercontent.com/renesas-rz/rzv_drp-ai_tvm/main/Dockerfile \
  -O "$CTX/Dockerfile"
cp "$TR" "$CTX/"

# Unzip the AI SDK and extract its Yocto toolchain installer (.sh) into context.
s=$(mktemp -d); TMPS+=("$s")
unzip -o -q "$ZIP" -d "$s"
# The Yocto toolchain installer is the big *toolchain*.sh (e.g.
# ai_sdk_setup/rz-vlp-...-rzv2h-evk-toolchain-5.0.11.sh). Pick the largest
# match so we don't grab a small board/flash helper script by mistake.
SDK_SH=$(find "$s" -iname "*toolchain*.sh" -printf '%s\t%p\n' | sort -rn | head -n1 | cut -f2-)
[[ -n "$SDK_SH" ]] || { echo "No toolchain .sh found inside $ZIP" >&2; exit 1; }
cp "$SDK_SH" "$CTX/"

echo "Build context ready in $CTX:"
ls -1 "$CTX"
echo
echo "Building image '$TAG' (PRODUCT=$PRODUCT) — builds the TVM fork, takes a while..."
docker build --build-arg PRODUCT="$PRODUCT" -t "$TAG" "$CTX"

cat <<EOF

Done. Start it with this repo mounted:
  docker run -it --rm -v "\$PWD/../..":/workspace/gst-python-ml $TAG bash
Inside, compile + run per this folder's README.md.
EOF
