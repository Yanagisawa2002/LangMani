#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: $0 --conda-prefix PATH <audit|execute|artifact-manifest> [args...]" >&2
  exit 2
}

if [[ "${1:-}" != "--conda-prefix" || -z "${2:-}" ]]; then
  usage
fi

conda_prefix="$2"
shift 2

if [[ ! -x "${conda_prefix}/bin/python" ]]; then
  echo "missing pinned Python: ${conda_prefix}/bin/python" >&2
  exit 2
fi
if [[ $# -lt 1 ]]; then
  usage
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
stage="$1"
shift

case "${stage}" in
  audit|execute|artifact-manifest) ;;
  *)
    echo "the external launcher cannot invoke internal child-probe directly" >&2
    exit 2
    ;;
esac

exec env -i \
  HOME=/root \
  PATH="${conda_prefix}/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
  CONDA_PREFIX="${conda_prefix}" \
  CONDA_DEFAULT_ENV="$(basename "${conda_prefix}")" \
  PYTHONNOUSERSITE=1 \
  PYTHONUNBUFFERED=1 \
  LANG=C.UTF-8 \
  LC_ALL=C.UTF-8 \
  PYTHONPATH="${repo_root}/src" \
  "${conda_prefix}/bin/python" \
  "${repo_root}/environment/run_v2_phase2b6_1r.py" \
  "${stage}" \
  --repo-root "${repo_root}" \
  --conda-prefix "${conda_prefix}" \
  "$@"
