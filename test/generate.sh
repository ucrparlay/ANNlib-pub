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

base["bigann10m"]="/data/zshen055/ANN/BIGANN/"
ds["bigann10m"]="${base["bigann10m"]}/base.100M.fbin:fbin"
query["bigann10m"]="${base["bigann10m"]}/query.10K.fbin:fbin"
gt["bigann10m"]="${base["bigann10m"]}/bigann-100M:ibin"

#base["cohere10m"]="/data/zshen055/ANN/cohere/cohere_large_10m"
base["cohere10m"]="data/cohere/cohere_large_10m"
ds["cohere10m"]="${base["cohere10m"]}/base.fbin:fbin"
query["cohere10m"]="${base["cohere10m"]}/query.fbin:fbin"
gt["cohere10m"]="${base["cohere10m"]}/gt.ibin:ibin"

base["deep10m"]="/data/zshen055/ANN/Yandex-DEEP"
ds["deep10m"]="${base["deep10m"]}/base.1B.fbin:fbin"
query["deep10m"]="${base["deep10m"]}/query.public.10K.fbin:fbin"
gt["deep10m"]="${base["deep10m"]}/deep-100M:ibin"

base["openai5m"]="/data/zshen055/ANN/openai/openai_large_5m"
ds["openai5m"]="${base["openai5m"]}/base.fbin:fbin"
query["openai5m"]="${base["openai5m"]}/query.fbin:fbin"
gt["openai5m"]="${base["openai5m"]}/gt.ibin:ibin"

set -x

name="bigann10m"
#./generate_index -n 10000000 -type float -dist L2 -out /data/zshen055/ANN/model/bigann10m_m32efc75_vr.model -in ${ds[$name]} -m 32 -efc 75 -alpha 0.85 

name="cohere10m"
#./generate_index -n 10000000 -type float -dist angular -out /data/zshen055/ANN/model/cohere10m_m32efc75_vr.model -in ${ds[$name]} -m 32 -efc 75 -alpha 0.85 
# ./genidx_ph -n 10000000 -type float -dist angular -out /tmp/model/cohere100k_ph.model -in ${ds[$name]} -m 32 -efc 75 -alpha 0.85 
# ./genidx_ol -n 10000000 -type float -dist angular -out /tmp/model/cohere100k_ol.model -in ${ds[$name]} -m 32 -efc 75 -alpha 0.85 
# ./genidx_hf -n 10000000 -type float -dist angular -out /tmp/model/cohere100k_hf.model -in ${ds[$name]} -m 32 -efc 75 -alpha 0.85 
./generate_index -n 100000 -type float -dist angular -out /tmp/model/cohere100k.model -in ${ds[$name]} -m 32 -efc 75 -alpha 0.85 

name="deep10m"
#./generate_index -n 10000000 -type float -dist angular -out /data/zshen055/ANN/model/deep10m_m32efc75_vr.model -in ${ds[$name]} -m 32 -efc 75 -alpha 0.85 

name="openai5m"
#./generate_index -n 5000000 -type float -dist angular -out /data/zshen055/ANN/model/openai5m_m32efc75_vr.model -in ${ds[$name]} -m 32 -efc 75 -alpha 0.85 
