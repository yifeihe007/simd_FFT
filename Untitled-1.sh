#!/bin/bash

for i in 32 64 128 256 512 1024
do
  for j in 32 64 128 256 512 1024
  do
	export NSAMP=i
    export NLOOP=j
	echo $NSAMP
    echo $NLOOP
done
done