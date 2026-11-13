#!/usr/bin/env bash
set -euo pipefail

package_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
workspace_dir="$(cd "${package_dir}/../.." && pwd)"
ros_edition="${1:-}"

case "${ros_edition}" in
  ROS1|ros1|1)
    requested_version=1
    default_distro=noetic
    ;;
  ROS2|ros2|2|humble)
    requested_version=2
    default_distro=humble
    ;;
  *)
    echo "Usage: $0 ROS1|ROS2" >&2
    exit 2
    ;;
esac

if [[ -n "${ROS_VERSION:-}" && "${ROS_VERSION}" != "${requested_version}" ]]; then
  echo "Sourced ROS_VERSION=${ROS_VERSION}, but ${ros_edition} was requested. Start a fresh shell." >&2
  exit 2
fi
if [[ -z "${ROS_VERSION:-}" ]]; then
  distro="${ROS_DISTRO:-${default_distro}}"
  setup="/opt/ros/${distro}/setup.bash"
  if [[ ! -f "${setup}" ]]; then
    echo "ROS setup not found: ${setup}" >&2
    exit 2
  fi
  set +u
  source "${setup}"
  set -u
fi
if [[ -n "${LIVOX_WS:-}" ]]; then
  set +u
  source "${LIVOX_WS%/}/setup.bash"
  set -u
fi

cd "${workspace_dir}"
if [[ "${requested_version}" == 1 ]]; then
  catkin_make -DCATKIN_WHITELIST_PACKAGES=fast_lio_sam -DFASTLIO_ROS_VERSION=1
else
  colcon build --packages-up-to fast_lio_sam --cmake-args -DFASTLIO_ROS_VERSION=2
fi
