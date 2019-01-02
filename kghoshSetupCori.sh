#!/bin/bash
module swap craype-haswell craype-mic-knl
export XTPE_LINK_TYPE=dynamic
export CRAYPE_LINK_TYPE=dynamic
./setup.sh
