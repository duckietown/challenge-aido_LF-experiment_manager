#!/bin/bash
input=$1
tmp1=$1.tmp.mp4
palette=$1.palette.png
output=$1.output.gif
ffmpeg -i $input -vf scale=160x120 -r 10 -y $tmp1
ffmpeg -i $tmp1 -vf palettegen=max_colors=256 -y $palette
ffmpeg -i $input  -i $palette -lavfi paletteuse=dither=bayer -y $output
