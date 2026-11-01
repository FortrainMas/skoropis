#!/usr/bin/env bash
# The whole GPU session from the laptop, unattended: upload code and data to a rented
# server, run scripts/server_run.sh there, wait, bring the results back. From Git Bash:
#
#   bash scripts/remote.sh root@1.2.3.4            # everything; PORT=2222 for another SSH port
#   bash scripts/remote.sh root@1.2.3.4 key        # only install the key (asks the password)
#   bash scripts/remote.sh root@1.2.3.4 status     # how far the run got
#   bash scripts/remote.sh root@1.2.3.4 fetch      # results only, e.g. after the laptop slept
#
# The server password is asked once, by ssh itself, to install this script's own key
# (~/.ssh/diploma_server, no passphrase); every later connection uses the key. The run
# lives on the server, so a sleeping laptop or a dropped connection does not stop it, and
# `fetch` picks the results up any time later. Running the script again skips what is done.
# Code goes as the working tree, uncommitted changes included; they are saved to
# logs/code.diff next to the run's logs, so the run stays reproducible.
set -euo pipefail
cd "$(dirname "$0")/.."

HOST=${1:?usage: bash scripts/remote.sh user@host [all|status|fetch]}
ACTION=${2:-all}
PORT=${PORT:-22}
RUN=${RUN:-scripts/server_run.sh}  # what to run on the server, relative to its copy of the repo
KEY=$HOME/.ssh/diploma_server
DIR=diploma  # the repo on the server, in the home directory
MODEL=models--Kansallisarkisto--cyrillic-large-handwritten
POLL=300     # seconds between checks while waiting

SSH_OPTS=(-o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30 -o ConnectTimeout=20)
KEY_OPTS=(-i "$KEY" -o IdentitiesOnly=yes -o BatchMode=yes)

say() { echo "=== $(date '+%F %T') $*"; }
remote() { ssh -p "$PORT" "${SSH_OPTS[@]}" "${KEY_OPTS[@]}" "$HOST" "$@"; }

keep_awake() {
  # Windows: tell the OS not to sleep while this script runs. The helper exits with the
  # script. Closing the lid can still put the laptop to sleep; the run itself goes on.
  command -v powershell.exe >/dev/null && [ -r /proc/$$/winpid ] || return 0
  local script="
    \$k = Add-Type -Name Awake -Namespace Diploma -PassThru -MemberDefinition '[DllImport(\"kernel32.dll\")] public static extern uint SetThreadExecutionState(uint flags);'
    [void]\$k::SetThreadExecutionState(0x80000001)  # ES_CONTINUOUS | ES_SYSTEM_REQUIRED
    while (Get-Process -Id $(cat /proc/$$/winpid) -ErrorAction SilentlyContinue) { Start-Sleep 30 }"
  powershell.exe -NoProfile -EncodedCommand "$(printf %s "$script" | iconv -t UTF-16LE | base64 -w0)" >/dev/null 2>&1 &
}

setup_key() {
  [ -f "$KEY" ] || ssh-keygen -q -t ed25519 -N "" -C diploma-server -f "$KEY"
  remote true 2>/dev/null && return
  say "installing an SSH key on $HOST: type the server password (once)"
  ssh -p "$PORT" "${SSH_OPTS[@]}" -o PubkeyAuthentication=no "$HOST" \
    'umask 077 && mkdir -p ~/.ssh && cat >> ~/.ssh/authorized_keys' <"$KEY.pub"
  remote true
}

ensure_tools() {
  remote 'command -v curl >/dev/null || { export DEBIAN_FRONTEND=noninteractive; apt-get update -q && apt-get install -yq curl; }
          command -v uv >/dev/null || test -x ~/.local/bin/uv || curl -LsSf https://astral.sh/uv/install.sh | sh'
}

ensure_driver() {
  remote nvidia-smi -L 2>/dev/null && return
  say "no NVIDIA driver on the server: installing the recommended one, then rebooting"
  remote 'set -e
    grep -q "^ID=ubuntu" /etc/os-release || { echo "not Ubuntu: install the NVIDIA driver by hand"; exit 1; }
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -q && apt-get install -yq ubuntu-drivers-common && ubuntu-drivers install'
  remote 'nohup reboot >/dev/null 2>&1 &' || true
  sleep 60
  local tries=0
  until remote true 2>/dev/null; do
    tries=$((tries + 1))
    [ "$tries" -lt 60 ] || { echo "the server did not come back after the reboot"; exit 1; }
    sleep 15
  done
  remote nvidia-smi -L
}

upload_code() {
  say "uploading code"
  { git ls-files -z --cached --others --exclude-standard; printf '.git\0'; } |
    tar --null --ignore-failed-read -T - -cf - | remote "mkdir -p $DIR && tar -xf - -C $DIR"
  git diff HEAD | remote "mkdir -p $DIR/logs && cat > $DIR/logs/code.diff"
}

upload_data() {
  if remote "test -f $DIR/data/.complete"; then say "data already on the server"; return; fi
  [ -f data/server-data.tar ] || tar -cf data/server-data.tar -C data strings digital-peter
  say "uploading data ($(du -h data/server-data.tar | cut -f1))"
  scp -P "$PORT" "${SSH_OPTS[@]}" "${KEY_OPTS[@]}" data/server-data.tar "$HOST:server-data.tar"
  remote "mkdir -p $DIR/data && tar -xf server-data.tar -C $DIR/data && rm server-data.tar && touch $DIR/data/.complete"
}

upload_model_if_needed() {
  # Without access to Hugging Face the model comes from the laptop's cache and the run goes offline.
  if remote 'curl -sfI --max-time 20 https://huggingface.co >/dev/null'; then return 1; fi
  say "Hugging Face is unreachable from the server: uploading the model from the local cache"
  if ! remote "test -d .cache/huggingface/hub/$MODEL"; then
    tar -cf - -C "$HOME/.cache/huggingface/hub" "$MODEL" |
      remote 'mkdir -p .cache/huggingface/hub && tar -xf - -C .cache/huggingface/hub'
  fi
}

running() { remote "test -f $DIR/logs/pid && kill -0 \$(cat $DIR/logs/pid) 2>/dev/null"; }

launch() {
  if running; then say "the run is already going"; return; fi
  local env="PATH=\$HOME/.local/bin:\$PATH"
  if upload_model_if_needed; then env="$env HF_HUB_OFFLINE=1"; fi
  remote "cd $DIR && rm -f logs/exit_code && export $env &&
    nohup setsid bash -c 'echo \$\$ > logs/pid; bash $RUN > logs/run.log 2>&1; echo \$? > logs/exit_code' \
      >/dev/null 2>&1 </dev/null &"
  say "started; on the server: tail -f ~/$DIR/logs/run.log"
}

progress() { remote "grep '^===' $DIR/logs/run.log 2>/dev/null | tail -${1:-1}" 2>/dev/null || echo "(server unreachable)"; }

wait_done() {
  say "waiting for the run (a check every $((POLL / 60)) min; Ctrl+C is safe, the run goes on)"
  until remote "test -f $DIR/logs/exit_code" 2>/dev/null; do
    sleep "$POLL"
    progress
    if ! running 2>/dev/null && ! remote "test -f $DIR/logs/exit_code" 2>/dev/null && remote true 2>/dev/null; then
      say "the run stopped without finishing (killed or rebooted?): bringing back what there is"
      return
    fi
  done
}

fetch() {
  say "bringing the results back"
  remote "cd $DIR && ls -d runs logs checkpoints/*/history.jsonl checkpoints/*/config.json checkpoints/crnn-*/best.pt \
    2>/dev/null | tar -cf - -T -" | tar -xf -
  [ ! -f logs/compare.md ] || cat logs/compare.md
  say "done: runs/, logs/, checkpoints/ (exit code of the run: $(cat logs/exit_code 2>/dev/null || echo none))"
  echo "Kansallisarkisto weights (checkpoints/trocr-*/best, 2.4 GB each) stay on the server."
}

case $ACTION in
  all)
    keep_awake
    setup_key
    ensure_tools
    ensure_driver
    upload_code
    upload_data
    launch
    wait_done
    fetch
    ;;
  key) setup_key && say "key installed: $HOST now accepts this laptop without a password" ;;
  status) progress 50 ;;
  fetch) fetch ;;
  *) echo "unknown action: $ACTION (all, key, status, fetch)" >&2; exit 2 ;;
esac
