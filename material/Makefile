CXX=g++
CXXFLAGS=-std=c++14 -Ofast -fopenmp -march=native
CPPFLAGS=-Wall -Wextra $(shell pkg-config cairo --cflags)
LDLIBS=-lcairo -lgd

N    = 11770
LAST = ${N}-cuda-offload3b-0837

animation.avi: ${LAST}.png
	mencoder -o "$@" mf://*.png -mf fps=60 -ovc x264 -x264encopts crf=11:preset=slower:frameref=5

${LAST}.png: timings-render ../timings_all.txt
	./timings-render

timings-render: timings-render.cc
	$(CXX) -o $@ $^ $(CXXFLAGS) $(CPPFLAGS) $(LDLIBS) $(LDFLAGS)

animation.gif: tile-${N}.gif
	rm -f tile-????[13579].gif
	gifsicle -O2 -o $@ -l0 -d3 -k128 tile-?????.gif

tile-${N}.gif: thu/${LAST}.png
	/chii/bisqwit/src/animmerger/animmerger \
		 --noalign -pc -Qq,32 --dc=2 --dm=2x2 -otile-%05d.gif thu/*.png \
	|| animmerger --noalign -pc -Qq,32 --dc=2 --dm=2x2 -otile-%05d.gif thu/*.png

thu/${LAST}.png: ${LAST}.png
	- mkdir thu
	/bin/ls ?????-*-*.png | xargs -P 8 -IX -n1 convert X -resize 640x360 thu/X
