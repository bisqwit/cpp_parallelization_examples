#!/bin/bash

# input filename
s="$1"

# 848,480
outw="$2"
outh="$3"

# -tiny
outfn="$4"

############################################

# default w and h as fallback
w=96
h=64
# find resolution
source <(mplayer -frames 0 -identify $s 2>/dev/null \
          | grep ID_VIDEO_.*I.*H.*= \
          | sed 's/ID_VIDEO_WIDTH/w/;s/ID_VIDEO_HEIGHT/h/')

# We now have $w and $h.

# Find out the width if converted to widescreen
ww=$((h*16/9))
#ww=$((h*256/135))
#if [ "$s" = "vid/e_008.avi"  ]; then ww=$w; fi
#if [ "$s" = "vid/e_009.avi"  ]; then ww=$w; fi
#if [ "$s" = "vid/e_010.avi"  ]; then ww=$w; fi

pad=""
if [ $w -lt $ww ]; then
        # Add padding if necessary
        pad="pad=$ww:$h:$(((ww-w)/2)):0,"
        w=$ww
fi

#w=$[w*$(ssh chii 'laske -q "ceil('$outw'/'$w')"')]
#h=$[h*$(ssh chii 'laske -q "ceil('$outh'/'$h')"')]

#w=$(php -r 'print '$w'*ceil('$outw'/'$w');')
#h=$(php -r 'print '$h'*ceil('$outh'/'$h');')

w=$(echo "scale=20; n=($outw/$w + 1-10^-6); scale=0; $w*(n/1)"|bc -l)
h=$(echo "scale=20; n=($outh/$h + 1-10^-6); scale=0; $h*(n/1)"|bc -l)

echo TRYING :  "$pad"scale="$w:$h"

trap "stty sane 2>/dev/null" SIGINT SIGTERM SIGILL
ffmpeg -an -f rawvideo -pix_fmt yuv444p -s "$w"x"$h" -r 60 \
	-i <( \
		ffmpeg -an -i "$s" -r 60 -pix_fmt yuv444p \
		       -sws_flags neighbor -vf "$pad"scale="$w:$h" \
		       -f rawvideo -y /dev/stdout \
	) \
        -sws_flags lanczos -r 60 \
        -s $outw"x"$outh -pix_fmt yuv444p -aspect 256/135 \
        -c:v h264 -threads 4 -preset superfast -refs 3 -trellis 0 \
        -x264-params 'subme=0' \
        -me_method dia -b-pyramid none -rc-lookahead 1 \
        -vf scale=$outw:$outh -sws_flags lanczos -r 60 -pix_fmt yuv444p -b:v 12000k \
        -y "$outfn"
s=$?
#-c:v ffv1
stty sane
exit $s
