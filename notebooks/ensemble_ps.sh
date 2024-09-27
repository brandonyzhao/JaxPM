#!/bin/bash

for i in $(seq 1 99);
do 
    echo $i
    python get_ps.py -d 4 -seed $i
done