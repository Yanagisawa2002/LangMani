#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: $0 --conda-prefix PATH <prepare|rendering-preflight|run|finalize|artifact-manifest> [args...]" >&2
  exit 2
}

if [[ "${1:-}" != "--conda-prefix" || -z "${2:-}" ]]; then
  usage
fi
conda_prefix="$2"
shift 2
if [[ ! -x "${conda_prefix}/bin/python" || $# -lt 1 ]]; then
  usage
fi

stage="$1"
shift
case "${stage}" in
  prepare|rendering-preflight|run|finalize|artifact-manifest) ;;
  *) usage ;;
esac

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
xdg_runtime_dir="$(mktemp -d /tmp/langmani-phase2b6-1-v2.XXXXXX)"
chmod 700 "${xdg_runtime_dir}"
cleanup() {
  case "${xdg_runtime_dir}" in
    /tmp/langmani-phase2b6-1-v2.*) rm -rf -- "${xdg_runtime_dir}" ;;
    *) echo "refusing unexpected XDG cleanup path: ${xdg_runtime_dir}" >&2 ;;
  esac
}
trap cleanup EXIT

env -i \
  HOME=/root \
  PATH="${conda_prefix}/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
  CONDA_PREFIX="${conda_prefix}" \
  CONDA_DEFAULT_ENV="$(basename "${conda_prefix}")" \
  PYTHONNOUSERSITE=1 \
  PYTHONUNBUFFERED=1 \
  LANG=C.UTF-8 \
  LC_ALL=C.UTF-8 \
  PYTHONPATH="${repo_root}/src" \
  VK_ICD_FILENAMES=/etc/vulkan/icd.d/my_nvidia_icd.json \
  __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json \
  CUDA_VISIBLE_DEVICES=0 \
  XDG_RUNTIME_DIR="${xdg_runtime_dir}" \
  "${conda_prefix}/bin/python" \
  "${repo_root}/environment/run_v2_phase2b6_1_v2.py" \
  "${stage}" \
  --repo-root "${repo_root}" \
  --adapter-launcher "${repo_root}/environment/run_v2_phase2b6_1_v2_forensic.sh" \
  "$@"
