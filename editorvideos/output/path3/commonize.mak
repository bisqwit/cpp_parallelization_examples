files := $(shell echo capture/inputter*.avi capture/e*.avi \
                 | xargs -L1 -d' ' echo | grep -v '\.avi-')

all:

define generate_rules
# params: 1=token, 2=width, 3=height, 4=output filename

all: $(4)

$(4): $(patsubst %.avi, %.avi-$(1).avi, $(files))
	mencoder -o "$(4)" -oac copy -ovc copy $$^

%.avi-$(1).avi: %.avi
	./commonize-helper.sh "$$<" "$(2)" "$(3)" "$$@"
endef

$(eval $(call generate_rules,tiny,848,480,vid_tiny.avi))
$(eval $(call generate_rules,resc,4096,2160,vid_resc.avi))
