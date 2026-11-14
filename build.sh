#!/usr/bin/env bash
set -euo pipefail

package_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
workspace_dir="$(cd "${package_dir}/../.." && pwd)"
ros_edition="${1:-${ROS_VERSION:-}}"

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
    echo "Source a ROS environment or use: $0 ROS1|ROS2" >&2
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
interface_dir="${package_dir}/interfaces/livox_ros_driver2"
interface_link="${workspace_dir}/src/livox_ros_driver2"
if [[ -L "${interface_link}" ]]; then
  if [[ "$(readlink -f "${interface_link}")" != "$(readlink -f "${interface_dir}")" ]]; then
    echo "Interface link points elsewhere: ${interface_link}" >&2
    exit 2
  fi
elif [[ ! -e "${interface_link}" ]]; then
  ln -s "${interface_dir}" "${interface_link}"
fi

cd "${workspace_dir}"
if [[ "${requested_version}" == 1 ]]; then
  catkin_make --source "${package_dir}/interfaces" \
    --build "${workspace_dir}/build/fastlio_interfaces_ros1" \
    --devel "${workspace_dir}/devel_fastlio_interfaces_ros1"
  set +u
  source "${workspace_dir}/devel_fastlio_interfaces_ros1/setup.bash"
  set -u
  catkin_make -DCATKIN_WHITELIST_PACKAGES=fast_lio_sam
else
  colcon build --base-paths "${package_dir}" "${interface_dir}"
fi
