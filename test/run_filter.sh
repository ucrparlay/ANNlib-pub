#!/bin/bash

export alpha=0.83
export batch=2
export ml=0.4
export k=10
export m=64     # HNSW
export R=96     # Filtered Vamana
export r=32     # Stitched Vamana Small
export SR=64    # Stitched Vamana Large
export efc=100
export rounds=1
export data_dir="/data/jsu068"

# make filter_test MODE=DEBUG -B &&
make filter_test -B &&

# set -x
mkdir -p results

### Synthetic filter label sets

## bigann
./filter_test -init 1000000 -step 1000000 -max 10000000 -type uint8 -dist L2 \
    -r ${r} -R ${R} -stitched_R ${SR} -m ${m} -ml ${ml} -efc ${efc} -alpha ${alpha} -b ${batch} -k ${k} -rounds ${rounds} \
    -beam 10,15,20,50,110,210,410,610 \
    -in ${data_dir}/bigann/bigann.128D.10M.euclidean.base.u8bin:u8bin \
    -q ${data_dir}/bigann/bigann.128D.10K.euclidean.query.u8bin:u8bin \
    -gts 3,10,15,20,37,50 \
    -lb ${data_dir}/bigann/bigann.10M.L50.zipf0.75.base.txt \
    &> "results/annlib.bigann.zipf.log"

# bigann.10M.L1000.random.base.txt - annlib.bigann.10M.k${k}.random1k.log
# bigann.10M.L10k.random.base.txt - annlib.bigann.10M.k${k}.random10k.log
# bigann.10M.L100k.random.base.txt - annlib.bigann.10M.k${k}.random100k.log
# bigann.10M.L50.zipf0.75.base.txt - annlib.bigann.10M.k${k}.log 
# 321,242,320,911,97 102 \\ 1,10,14,20,49,100,505,994,1495,1983 \\ 3,4,6,8,10,14,20,49,100,200,218,249,505,994
# 10,15,20,30,50,70,90,110,130,150,170,190,210,230,250,270,290,310,330,350,370,390,410,430,450,470,490,510,530,550,570,590,610,630,650
# -lb ${data_dir}/bigann/bigann.10M.L2000.zipf.base.txt \
# .add: zipf0.75 addition, .zipf: zipf0.75 new specificity

## deep
./filter_test -init 1000000 -step 1000000 -max 10000000 -type float -dist angular \
    -r ${r} -R ${R} -stitched_R ${SR} -m ${m} -ml ${ml} -efc ${efc} -alpha ${alpha} -b ${batch} -k ${k} -rounds ${rounds} \
    -beam 10,15,20,50,110,210,410,610 \
    -in ${data_dir}/deep/deep.96D.10M.angular.base.fbin:fbin \
    -q ${data_dir}/deep/deep.96D.10K.angular.query.fbin:fbin \
    -gts 3,10,15,20,37,50 \
    -lb ${data_dir}/deep/bigann.10M.L50.zipf0.75.base.txt \
    &> "results/annlib.deep.zipf.log"

#     -gts 1,20,49,100,250,501,753,996 \\ 4,6,8,10,20,49,75,100,150,200,388,501,753,996 \\ 1,4,16,50
# -lb ${data_dir}/deep/deep.10M.L1000.zipf.base.txt \

## cohere
./filter_test -init 1000000 -step 1000000 -max 10000000 -type float -dist angular \
    -r ${r} -R ${R} -stitched_R ${SR} -m ${m} -ml ${ml} -efc ${efc} -alpha ${alpha} -b ${batch} -k ${k} -rounds ${rounds} \
    -beam 10,15,20,50,110,210,410,610 \
    -in ${data_dir}/cohere/cohere.768D.10M.angular.base.fbin:fbin \
    -q ${data_dir}/cohere/cohere.768D.10M.angular.query.fbin:fbin \
    -gts 3,10,15,20,37,50 \
    -lb ${data_dir}/cohere/cohere.10M.L50.zipf0.75.txt  \
    &> "results/annlib.cohere.zipf.log"

## openai
./filter_test -init 1000000 -step 1000000 -max 5000000 -type float -dist angular \
    -r ${r} -R ${R} -stitched_R ${SR} -m ${m} -ml ${ml} -efc ${efc} -alpha ${alpha} -b ${batch} -k ${k} -rounds ${rounds} \
    -beam 10,15,20,50,110,210,410,610 \
    -in ${data_dir}/openai/openai.1536D.5M.angular.base.fbin:fbin \
    -q ${data_dir}/openai/openai.1536D.5M.angular.query.fbin:fbin \
    -gts 3,10,15,20,37,50 \
    -lb ${data_dir}/openai/openai.5M.L50.zipf0.75.txt  \
    &> "results/annlib.openai.zipf.log"


### Natural filter label sets

## marco
./filter_test -init 1000000 -step 1000000 -max 10000000 -type float -dist L2 \
    -r ${r} -R ${R} -stitched_R ${SR} -m ${m} -ml ${ml} -efc ${efc} -alpha ${alpha} -b ${batch} -k ${k} -rounds ${rounds} \
    -beam 10,15,20,50,110,210,410,610 \
    -in ${data_dir}/marco/embedding/marco.768D.10M.euclidean.fbin:fbin \
    -q ${data_dir}/marco/query/marco.768D.10K.euclidean.fbin:fbin \
    -gts 48,49,50,51,52 \
    -lb ${data_dir}/marco/embedding/marco.filter.base.10M.new.txt \
    &> "results/annlib.marco.zipf.log"

## yfcc
./filter_test -init 1000000 -step 1000000 -max 10000000 -type uint8 -dist L2 \
    -r ${r} -R ${R} -stitched_R ${SR} -m ${m} -ml ${ml} -efc ${efc} -alpha ${alpha} -b ${batch} -k ${k} -rounds ${rounds} \
    -beam 10,15,20,50,110,210,410,610 \
    -in ${data_dir}/yfcc/yfcc.192D.10M.euclidean.base.u8bin:u8bin \
    -q ${data_dir}/yfcc/yfcc.192D.100K.euclidean.query.u8bin:u8bin \
    -gts 29,17,28,24,106,0 \
    -lb ${data_dir}/yfcc/yfcc.filter.base.txt \
    &> "results/annlib.yfcc.zipf.log"
