#!/bin/sh
# Open an OpenFOAM case of this project in ParaView.
#
#   ./view_in_paraview.sh              -> unit_cell (flow: U, p, k, omega, nut)
#   ./view_in_paraview.sh thermal      -> thermal case (the 8 temperature fields)
#   ./view_in_paraview.sh mesh_study/fine
#
# paraFoam must be run from inside a case folder (it looks for ./constant);
# running it anywhere else gives "FATAL ERROR: Mesh constant does not exist".
# This script changes into the right folder first, so it works from anywhere.
#
# If ParaView crashes on start ("Loguru caught a signal: SIGSEGV") - common
# on WSL, where the graphics (OpenGL) driver is the usual cause - try, in order:
#   ./view_in_paraview.sh unit_cell --software  # software OpenGL rendering
#   ./view_in_paraview.sh unit_cell -builtin    # ParaView's own OpenFOAM reader
#   ./view_in_paraview.sh unit_cell --windows   # open it in Windows ParaView instead
# --windows only creates <case>.foam and prints the path to open from Windows.
# Other extra arguments are passed on to paraFoam.

here=$(cd "$(dirname "$0")" && pwd)
case_name=${1:-unit_cell}
[ $# -gt 0 ] && shift
case_dir="$here/$case_name"

if [ ! -d "$case_dir" ]; then
    echo "No case folder $case_dir"
    echo "Cases here: $(cd "$here" && ls -d unit_cell thermal mesh_study/* 2>/dev/null | tr '\n' ' ')"
    exit 1
fi
if [ ! -d "$case_dir/constant/polyMesh" ]; then
    echo "$case_dir has no mesh yet (constant/polyMesh) - run its Allrun first."
    exit 1
fi

windows=0
args=""
for a in "$@"; do
    case "$a" in
        --windows)  windows=1 ;;
        --software) export LIBGL_ALWAYS_SOFTWARE=1 GALLIUM_DRIVER=llvmpipe ;;
        *)          args="$args $a" ;;
    esac
done

if [ $windows -eq 1 ]; then
    foam_file="$case_dir/$(basename "$case_dir").foam"
    touch "$foam_file"
    echo "Created $foam_file"
    if command -v wslpath >/dev/null 2>&1; then
        echo "In Windows ParaView: File -> Open, paste this path, then click Apply:"
        echo "    $(wslpath -w "$foam_file")"
    else
        echo "Open this file in ParaView (File -> Open), then click Apply."
    fi
    exit 0
fi

if ! command -v paraFoam >/dev/null 2>&1; then
    echo "paraFoam not found - load OpenFOAM first:  source /opt/openfoam13/etc/bashrc"
    exit 1
fi

cd "$case_dir" || exit 1
echo "Opening $case_dir in ParaView..."
# shellcheck disable=SC2086
exec paraFoam $args
