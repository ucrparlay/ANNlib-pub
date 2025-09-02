#!/usr/bin/env bash

RED='\033[0;31m'
NC='\033[0m'

protect(){
	name=$1

	dir=$(dirname ${name})
	if [ ! -d ${dir} ]; then
		echo -e "${RED}create ${dir}${NC}" >&2
		mkdir -p ${dir}
	fi

	local alt_name=${name}
	if [[ -e ${name} ]]; then
		i=0
		alt_name=${name}.${i}
		while [[ -e ${alt_name} ]]; do
			let i++
			alt_name=${name}.${i}
		done
		echo -e "${RED}using the alternative name ${alt_name}${NC}" >&2
	fi
	echo "$alt_name"
}

declare -A base
declare -A ds
declare -A query
declare -A gt
declare -A gtf

base["bigann"]="/data/zshen055/ANN/BIGANN/"
ds["bigann"]="${base["bigann"]}/base.100M.fbin:fbin"
query["bigann"]="${base["bigann"]}/query.10K.fbin:fbin"
gt["bigann"]="${base["bigann"]}/bigann-100M:ibin"
gtf["bigann"]="${base["bigann"]}/gt/"

base["cohere"]="/data/zshen055/ANN/cohere/cohere_large_10m"
ds["cohere"]="${base["cohere"]}/base.fbin:fbin"
query["cohere"]="${base["cohere"]}/query.fbin:fbin"
gt["cohere"]="${base["cohere"]}/gt.ibin:ibin"
gtf["cohere"]="${base["cohere"]}/gt/"

base["deep"]="/data/zshen055/ANN/Yandex-DEEP"
ds["deep"]="${base["deep"]}/base.1B.fbin:fbin"
query["deep"]="${base["deep"]}/query.public.10K.fbin:fbin"
gt["deep"]="${base["deep"]}/deep-100M:ibin"
gtf["deep"]="${base["deep"]}/gt/"

base["openai"]="/data/zshen055/ANN/openai/openai_large_5m"
ds["openai"]="${base["openai"]}/base.fbin:fbin"
query["openai"]="${base["openai"]}/query.fbin:fbin"
gt["openai"]="${base["openai"]}/gt.ibin:ibin"
gtf["openai"]="${base["openai"]}/gt/"

name="bigann"
./dyn_test -init 100000000 -step 10000000 -max 100000000 -type float -dist L2 -ml 0.36 -m 50 -efc 100 -alpha 0.85 -b 2 -f 0 -in ${ds[$name]} -q ${query[$name]} -gtf ${gtf[$name]}/100M -efs 10,20,50,100,200,300,400,500 -k 10 -w 0 -le 1 > $(protect results/0722-final-query/bigann-100M-10M.log)
./dyn_test -init 10000000 -step 1000000 -max 10000000 -type float -dist L2 -ml 0.36 -m 50 -efc 100 -alpha 0.85 -b 2 -f 0 -in ${ds[$name]} -q ${query[$name]} -gtf ${gtf[$name]} -efs 10,20,50,100,200,300,400,500 -k 10 -w 0 -le 1 > $(protect results/0722-final-query/bigann-10M-1M.log)

name="cohere"
./dyn_test -init 10000000 -step 1000000 -max 10000000 -type float -dist angular -ml 0.36 -m 50 -efc 100 -alpha 0.85 -b 2 -f 0 -in ${ds[$name]} -q ${query[$name]} -gtf ${gtf[$name]} -efs 10,20,50,100,200,300,400,500 -k 10 -w 0 -le 1 > $(protect results/0722-final-query/cohere-10M-1M.log)

name="deep"
./dyn_test -init 100000000 -step 10000000 -max 100000000 -type float -dist angular -ml 0.36 -m 50 -efc 100 -alpha 0.85 -b 2 -f 0 -in ${ds[$name]} -q ${query[$name]} -gtf ${gtf[$name]}/100M -efs 10,20,50,100,200,300,400,500 -k 10 -w 0 -le 1 > $(protect results/0722-final-query/deep-100M-10M.log)
./dyn_test -init 10000000 -step 1000000 -max 10000000 -type float -dist angular -ml 0.36 -m 50 -efc 100 -alpha 0.85 -b 2 -f 0 -in ${ds[$name]} -q ${query[$name]} -gtf ${base[$name]}/deep-10M:ibin -efs 10,20,50,100,200,300,400,500 -k 10 -w 0 -le 1 > $(protect results/0722-final-query/deep-10M-1M.log)

name="openai"
./dyn_test -init 5000000 -step 500000 -max 5000000 -type float -dist angular -ml 0.36 -m 50 -efc 100 -alpha 0.85 -b 2 -f 0 -in ${ds[$name]} -q ${query[$name]} -gtf ${gtf[$name]} -efs 10,20,50,100,200,300,400,500 -k 10 -w 0 -le 1 > $(protect results/0722-final-query/openai-5M-500K.log)
