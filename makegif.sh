#!/bin/bash
input=$1
tmp1=$1.tmp.mp4
palette=$1.palette.tmp.png
output=$2


# ffmpeg -i $input -vf palettegen=max_colors=64 -y $palette
ffmpeg -i $input -vf palettegen -y $palette

ffmpeg -i $input  -i $palette -lavfi paletteuse=dither=bayer  -y $output
rm -f $tmp1 $palette
