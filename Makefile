SRC = protobuffs

JAVA_OUT_PATH = ./keystone_pipeline/src/main/java/protobuffs
JAVA_OUT_FILE = Features.java
JAVA_OUT = $(JAVA_OUT_PATH)/$(JAVA_OUT_FILE)


PYTHON_OUT_PATH = .
PYTHON_OUT_FILE = features_pb2.py

PYTHON_OUT = $(PYTHON_OUT_PATH)/$(PYTHON_OUT_FILE)

protobuffs: $(PYTHON_OUT) $(JAVA_OUT)

$(PYTHON_OUT): $(SRC)/features.proto
	protoc -I=$(SRC) --python_out=$(PYTHON_OUT_PATH) $<

$(JAVA_OUT): $(SRC)/features.proto
	protoc -I=$(SRC) --java_out=$(JAVA_OUT_PATH) $<

