#!/usr/bin/env bash
#
# r2.sh - upload/download job files to a Cloudflare R2 bucket via rclone.
#
# This script contains NO secrets and is safe to commit. Credentials are read
# from a separate file (default: .r2_credentials next to this script), which is
# gitignored. On first use the script prompts for them and writes that file.
#
# Requires: rclone (https://rclone.org). If it's missing, this script tells you
# exactly how to install it.
#
# Usage:
#   ./r2.sh up   <local-file> [remote-name]   upload a file (remote-name defaults to basename)
#   ./r2.sh down <remote-name> [local-path]   download a file (local-path defaults to ./basename)
#   ./r2.sh ls   [prefix]                      list objects in the bucket
#   ./r2.sh rm   <remote-name>                 delete one object
#   ./r2.sh setup                              (re)enter credentials
#
# Examples:
#   ./r2.sh up archive_100.tar.zst            # local box: push the matrix bundle
#   ./r2.sh down archive_100.tar.zst          # rented box: pull it back
#   ./r2.sh up msieve.dat.dep dep_12345       # push a result under a chosen name

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CREDS_FILE="${R2_CREDENTIALS_FILE:-$SCRIPT_DIR/.r2_credentials}"

# rclone transfer tuning: parallelize chunks of a single large file so one big
# .mat/.tar doesn't bottleneck on a single stream. Safe defaults; override with
# R2_RCLONE_OPTS if you want.
RCLONE_OPTS=(
    --progress
    --s3-chunk-size 64M
    --s3-upload-concurrency 8
    --multi-thread-streams 8
    --multi-thread-cutoff 256M
)
# shellcheck disable=SC2206
[ -n "${R2_RCLONE_OPTS:-}" ] && RCLONE_OPTS+=(${R2_RCLONE_OPTS})

die() { echo "error: $*" >&2; exit 1; }

usage() {
    # print the leading comment block (skip the shebang, stop at first non-# line)
    awk 'NR==1{next} /^#/{sub(/^# ?/,""); print; next} {exit}' "${BASH_SOURCE[0]}"
    exit "${1:-0}"
}

require_rclone() {
    command -v rclone >/dev/null 2>&1 && return 0
    cat >&2 <<'EOF'
error: rclone is not installed (this script needs it for R2 transfers).

Quick install (works on most Linux rented instances):
    curl https://rclone.org/install.sh | sudo bash

Distro packages:
    Debian/Ubuntu:   sudo apt-get update && sudo apt-get install -y rclone
    Fedora/RHEL:     sudo dnf install -y rclone
    Arch:            sudo pacman -S rclone
    macOS:           brew install rclone

Static binaries / other platforms:  https://rclone.org/downloads/

Then re-run this script.
EOF
    exit 1
}

setup_creds() {
    echo "Configuring R2 credentials -> $CREDS_FILE"
    echo "(find these in the Cloudflare dashboard: R2 -> Manage R2 API Tokens,"
    echo " and your Account ID on the R2 overview page)"
    echo
    local account_id access_key secret_key bucket
    read -r  -p "R2 Account ID:        " account_id
    read -r  -p "R2 Access Key ID:     " access_key
    read -rs -p "R2 Secret Access Key: " secret_key; echo
    read -r  -p "R2 Bucket name:       " bucket

    [ -n "$account_id" ] && [ -n "$access_key" ] && [ -n "$secret_key" ] && [ -n "$bucket" ] \
        || die "all four values are required"

    local umask_old; umask_old="$(umask)"
    umask 077                       # create the file unreadable to others
    cat > "$CREDS_FILE" <<EOF
# Cloudflare R2 credentials for r2.sh - DO NOT COMMIT.
R2_ACCOUNT_ID=$account_id
R2_ACCESS_KEY_ID=$access_key
R2_SECRET_ACCESS_KEY=$secret_key
R2_BUCKET=$bucket
EOF
    umask "$umask_old"
    chmod 600 "$CREDS_FILE"
    echo "Saved. (chmod 600; keep this file off git.)"
}

load_creds() {
    [ -f "$CREDS_FILE" ] || { setup_creds; }
    # shellcheck disable=SC1090
    source "$CREDS_FILE"
    : "${R2_ACCOUNT_ID:?missing R2_ACCOUNT_ID in $CREDS_FILE}"
    : "${R2_ACCESS_KEY_ID:?missing R2_ACCESS_KEY_ID in $CREDS_FILE}"
    : "${R2_SECRET_ACCESS_KEY:?missing R2_SECRET_ACCESS_KEY in $CREDS_FILE}"
    : "${R2_BUCKET:?missing R2_BUCKET in $CREDS_FILE}"

    # Define an on-the-fly rclone remote named "r2" purely from env vars,
    # so no rclone config file is needed - just copy this script + creds file.
    export RCLONE_CONFIG_R2_TYPE=s3
    export RCLONE_CONFIG_R2_PROVIDER=Cloudflare
    export RCLONE_CONFIG_R2_REGION=auto
    export RCLONE_CONFIG_R2_ACCESS_KEY_ID="$R2_ACCESS_KEY_ID"
    export RCLONE_CONFIG_R2_SECRET_ACCESS_KEY="$R2_SECRET_ACCESS_KEY"
    export RCLONE_CONFIG_R2_ENDPOINT="https://${R2_ACCOUNT_ID}.r2.cloudflarestorage.com"
    export RCLONE_CONFIG_R2_NO_CHECK_BUCKET=true   # scoped tokens can't list/create buckets
}

cmd="${1:-}"; shift || true
case "$cmd" in
    up)
        [ $# -ge 1 ] || usage 1
        require_rclone; load_creds
        src="$1"; [ -f "$src" ] || die "no such file: $src"
        name="${2:-$(basename "$src")}"
        echo ">> uploading $src -> r2:$R2_BUCKET/$name"
        rclone "${RCLONE_OPTS[@]}" copyto "$src" "r2:$R2_BUCKET/$name"
        ;;
    down)
        [ $# -ge 1 ] || usage 1
        require_rclone; load_creds
        name="$1"; dest="${2:-./$(basename "$name")}"
        echo ">> downloading r2:$R2_BUCKET/$name -> $dest"
        rclone "${RCLONE_OPTS[@]}" copyto "r2:$R2_BUCKET/$name" "$dest"
        ;;
    ls)
        require_rclone; load_creds
        rclone lsl "r2:$R2_BUCKET/${1:-}"
        ;;
    rm)
        [ $# -ge 1 ] || usage 1
        require_rclone; load_creds
        echo ">> deleting r2:$R2_BUCKET/$1"
        rclone deletefile "r2:$R2_BUCKET/$1"
        ;;
    setup)
        setup_creds
        ;;
    ""|-h|--help|help)
        usage 0
        ;;
    *)
        die "unknown command: $cmd (try: up | down | ls | rm | setup)"
        ;;
esac
