#!/bin/bash

echo "Compiling FlatBuffers schema..."

flatc --cpp -o include/ nerf_messages.fbs

flatc --python nerf_messages.fbs

echo "Compiled successfully!"
