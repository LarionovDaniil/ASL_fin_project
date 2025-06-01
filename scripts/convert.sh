#!/bin/bash
onnx_file="outputs/model.onnx"
engine_file="outputs/model.engine"
trtexec --onnx=$onnx_file --saveEngine=$engine_file --fp16
