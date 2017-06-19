CXX=g++
CXXFLAGS=-std=c++14 -Ofast -fopenmp -march=native
CPPFLAGS=-Wall -Wextra $(shell pkg-config cairo --cflags)
LDLIBS=-lcairo -lgd

animation.avi: 11772-cuda-offload3b-0839.png
	mencoder -o "$@" mf://*.png -mf fps=60 -ovc x264 -x264encopts crf=11:preset=slower:frameref=5

11772-cuda-offload3b-0839.png: timings-render ../timings_all.txt
	./timings-render

timings-render: timings-render.cc
	$(CXX) -o $@ $^ $(CXXFLAGS) $(CPPFLAGS) $(LDLIBS) $(LDFLAGS)
