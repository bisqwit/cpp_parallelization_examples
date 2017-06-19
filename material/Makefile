CXX=g++
CXXFLAGS=-std=c++14 -Ofast -fopenmp -march=native
CPPFLAGS=-Wall -Wextra $(shell pkg-config cairo --cflags)
LDLIBS=-lcairo -lgd

N    = 11772
LAST = ${N}-cuda-offload3b-0839

animation.avi: ${LAST}.png
	mencoder -o "$@" mf://*.png -mf fps=60 -ovc x264 -x264encopts crf=11:preset=slower:frameref=5

${LAST}.png: timings-render ../timings_all.txt
	./timings-render

timings-render: timings-render.cc
	$(CXX) -o $@ $^ $(CXXFLAGS) $(CPPFLAGS) $(LDLIBS) $(LDFLAGS)

animation.gif: tile-${N}.gif
	gifsicle -O2 -o $@ -l0 -d3 -k128 tile-?????.gif

tile-${N}.gif: thu/${LAST}.png
	animmerger --noalign -pc -Qd,64 --dc=4 --dm=4x4 -otile-%05d.gif thu/*.png

thu/${LAST}.png: ${LAST}.png
	- mkdir thu
	/bin/ls ?????-*-*.png | xargs -P 8 -IX -n1 convert X -resize 424x240 -sharpen 1 thu/X
