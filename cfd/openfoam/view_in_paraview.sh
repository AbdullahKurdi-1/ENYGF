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
# Extra arguments are passed on to paraFoam (e.g. -builtin, see below).

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
if ! command -v paraFoam >/dev/null 2>&1; then
    echo "paraFoam not found - load OpenFOAM first:  source /opt/openfoam13/etc/bashrc"
    exit 1
fi

cd "$case_dir" || exit 1
echo "Opening $case_dir in ParaView..."
# If ParaView complains that the OpenFOAM reader module is missing, run again
# with -builtin (ParaView's own reader):  ./view_in_paraview.sh unit_cell -builtin
exec paraFoam "$@"
