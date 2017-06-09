CXX=g++
CXXFLAGS=-std=c++14 -Ofast -fopenmp
CPPFLAGS=-Wall -Wextra $(shell pkg-config cairo --cflags)
LDLIBS=-lcairo -lgd

animation.avi: 4199-explicit-simd-0839.png
	mencoder -o "$@" mf://*.png -mf fps=60 -ovc x264 -x264encopts crf=11:preset=slower:frameref=5

4199-explicit-simd-0839.png: timings-render ../timings_all.txt
	./timings-render

timings-render: timings-render.cc
	$(CXX) -o $@ $^ $(CXXFLAGS) $(CPPFLAGS) $(LDLIBS) $(LDFLAGS)
