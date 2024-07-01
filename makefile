SHELL:=/bin/bash
TARGETS=Create_dirs , Preprocess , Train

.PHONY: all
all: $(TARGETS)

# 定义需要创建的目录变量
DIRS = \
    Osiris_file \
    Osiris_file/probe1_top_bottom_0_15 \
    Osiris_file/probe1_top_bottom_0_15/dataset \
    Osiris_file/probe1_top_bottom_0_15/parameter \
    Osiris_file/probe1_top_bottom_0_15/result \
    Osiris_file/probe2_lateral_28_51 \
    Osiris_file/probe2_lateral_28_51/dataset \
    Osiris_file/probe2_lateral_28_51/parameter \
    Osiris_file/probe2_lateral_28_51/result \
    Osiris_file/probe3_lateral_bias_16_27_and_52_63 \
    Osiris_file/probe3_lateral_bias_16_27_and_52_63/dataset \
    Osiris_file/probe3_lateral_bias_16_27_and_52_63/parameter \
    Osiris_file/probe3_lateral_bias_16_27_and_52_63/result

# 目标：创建所有目录
Create_dirs:
	@$(foreach dir,$(DIRS),mkdir -p $(dir);)