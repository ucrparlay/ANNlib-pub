#!/usr/bin/env bash

RED='\033[0;31m'
NC='\033[0m'

protect(){
	name=$1

	dir=$(dirname ${name})
	if [ ! -d ${dir} ]; then
		echo -e "${RED}create ${dir}${NC}"
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
		echo -e "${RED}using the alternative name ${alt_name}${NC}"
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

base["cohere"]="/data/zshen055/ANN/cohere/cohere_large_10m"
ds["cohere"]="${base["cohere"]}/base.fbin:fbin"
query["cohere"]="${base["cohere"]}/query.fbin:fbin"
gt["cohere"]="${base["cohere"]}/gt.ibin:ibin"

base["deep"]="/data/zshen055/ANN/Yandex-DEEP"
ds["deep"]="${base["deep"]}/base.1B.fbin:fbin"
query["deep"]="${base["deep"]}/query.public.10K.fbin:fbin"
gt["deep"]="${base["deep"]}/deep-100M:ibin"

base["openai"]="/data/zshen055/ANN/openai/openai_large_5m"
ds["openai"]="${base["openai"]}/base.fbin:fbin"
query["openai"]="${base["openai"]}/query.fbin:fbin"
gt["openai"]="${base["openai"]}/gt.ibin:ibin"

name="bigann"
mkdir "${base[$name]}/gt/" 
./generate_groundtruth -init 1000000 -step 1000000 -max 100000000 -type float -dist L2 -in ${ds[$name]} -q ${query[$name]} -gtf "${base[$name]}/gt/100M/" -k 200
./generate_groundtruth -init 1000000 -step 1000000 -max 10000000 -type float -dist L2 -in ${ds[$name]} -q ${query[$name]} -gtf "${base[$name]}/gt/" -k 200

name="cohere"
mkdir "${base[$name]}/gt/" 
./generate_groundtruth -init 1000000 -step 1000000 -max 10000000 -type float -dist angular -in ${ds[$name]} -q ${query[$name]} -gtf "${base[$name]}/gt/" -k 200

name="deep"
mkdir "${base[$name]}/gt/" 
./generate_groundtruth -init 1000000 -step 1000000 -max 100000000 -type float -dist angular -in ${ds[$name]} -q ${query[$name]} -gtf "${base[$name]}/gt/100M/" -k 200
./generate_groundtruth -init 1000000 -step 1000000 -max 10000000 -type float -dist angular -in ${ds[$name]} -q ${query[$name]} -gtf "${base[$name]}/gt/" -k 200

name="openai"
mkdir "${base[$name]}/gt/" 
./generate_groundtruth -init 500000 -step 500000 -max 5000000 -type float -dist angular -in ${ds[$name]} -q ${query[$name]} -gtf "${base[$name]}/gt/" -k 200
