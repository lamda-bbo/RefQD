ENV_NAME = refqd
ENV_YML_PATH = environment.yml
ENVPOOL_WHL_PATH = .external/envpool-0.8.4.1-cp311-cp311-linux_x86_64.whl
ENVPOOL_SCRIPT = .external/envpool.sh

.PHONY : create update remove dry

create: $(ENVPOOL_WHL_PATH)
	conda env create -f '$(ENV_YML_PATH)' -n '$(ENV_NAME)'

update: $(ENVPOOL_WHL_PATH)
	conda env update -f '$(ENV_YML_PATH)' -n '$(ENV_NAME)' --prune

remove:
	conda env remove -n '$(ENV_NAME)'

$(ENVPOOL_WHL_PATH):
	bash '$(ENVPOOL_SCRIPT)'
