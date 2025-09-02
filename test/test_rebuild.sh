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

base["bigann10m"]="/data/zshen055/ANN/BIGANN/"
ds["bigann10m"]="${base["bigann10m"]}/base.100M.fbin:fbin"
query["bigann10m"]="${base["bigann10m"]}/query.10K.fbin:fbin"
gt["bigann10m"]="${base["bigann10m"]}/bigann-10M:ibin"
gtf["bigann10m"]="${base["bigann10m"]}/gt/"

base["bigann100m"]="/data/zshen055/ANN/BIGANN/"
ds["bigann100m"]="${base["bigann100m"]}/base.100M.fbin:fbin"
query["bigann100m"]="${base["bigann100m"]}/query.10K.fbin:fbin"
gt["bigann100m"]="${base["bigann100m"]}/bigann-100M:ibin"
gtf["bigann100m"]="${base["bigann100m"]}/gt/"

base["cohere10m"]="/data/zshen055/ANN/cohere/cohere_large_10m"
ds["cohere10m"]="${base["cohere10m"]}/base.fbin:fbin"
query["cohere10m"]="${base["cohere10m"]}/query.fbin:fbin"
gt["cohere10m"]="${base["cohere10m"]}/gt.ibin:ibin"
gtf["cohere10m"]="${base["cohere10m"]}/gt/"

base["deep10m"]="/data/zshen055/ANN/Yandex-DEEP"
ds["deep10m"]="${base["deep10m"]}/base.1B.fbin:fbin"
query["deep10m"]="${base["deep10m"]}/query.public.10K.fbin:fbin"
gt["deep10m"]="${base["deep10m"]}/deep-10M:ibin"
gtf["deep10m"]="${base["deep10m"]}/gt/"

base["deep100m"]="/data/zshen055/ANN/Yandex-DEEP"
ds["deep100m"]="${base["deep100m"]}/base.1B.fbin:fbin"
query["deep100m"]="${base["deep100m"]}/query.public.10K.fbin:fbin"
gt["deep100m"]="${base["deep100m"]}/deep-10M:ibin"
gtf["deep100m"]="${base["deep100m"]}/gt/"

base["openai5m"]="/data/zshen055/ANN/openai/openai_large_5m"
ds["openai5m"]="${base["openai5m"]}/base.fbin:fbin"
query["openai5m"]="${base["openai5m"]}/query.fbin:fbin"
gt["openai5m"]="${base["openai5m"]}/gt.ibin:ibin"
gtf["openai5m"]="${base["openai5m"]}/gt/"

name="bigann10m"
./test_rebuild -init 1000000 -step 1000000 -max 10000000 -type float -dist L2  -ml 0.36 -m 32 -efc 75 -alpha 0.85 -b 2 -f 0 -in ${ds[$name]} -q ${query[$name]} -g ${gt[$name]} -efs 10,20,50,100,200,300,400,500 -k 10 -w 0 -le 1 -gtf ${gtf[$name]} > $(protect results/0715-beamSearch2/rebuild/bigann-10M.log)
name="bigann100m"
./test_rebuild -init 10000000 -step 10000000 -max 100000000 -type float -dist L2  -ml 0.36 -m 32 -efc 75 -alpha 0.85 -b 2 -f 0 -in ${ds[$name]} -q ${query[$name]} -g ${gt[$name]} -efs 10,20,50,100,200,300,400,500 -k 10 -w 0 -le 1 -gtf ${gtf[$name]}/100M/ > $(protect results/0715-beamSearch2/rebuild/bigann-100M.log)

name="cohere10m"
./test_rebuild -init 1000000 -step 1000000 -max 10000000 -type float -dist angular  -ml 0.36 -m 32 -efc 75 -alpha 0.85 -b 2 -f 0 -in ${ds[$name]} -q ${query[$name]} -g ${gt[$name]} -efs 10,20,50,100,200,300,400,500 -k 10 -w 0 -le 1 -gtf ${gtf[$name]} > $(protect results/0715-beamSearch2/rebuild/cohere-10M.log)
#./dt_origin -init 1000000 -step 1000000 -max 10000000 -type float -dist angular  -ml 0.36 -m 32 -efc 75 -alpha 0.85 -b 2 -f 0 -in ${ds[$name]} -q ${query[$name]} -g ${gt[$name]} -efs 10,20,50,100,200,300,400,500 -k 10 -w 0 -le 1 -gtf ${gtf[$name]} > $(protect results/0715-beamSearch2/rebuild/origin.log)
#./dt_second -init 1000000 -step 1000000 -max 10000000 -type float -dist angular  -ml 0.36 -m 32 -efc 75 -alpha 0.85 -b 2 -f 0 -in ${ds[$name]} -q ${query[$name]} -g ${gt[$name]} -efs 10,20,50,100,200,300,400,500 -k 10 -w 0 -le 1 -gtf ${gtf[$name]} > $(protect results/0715-beamSearch2/rebuild/second.log)
#./dt_occlude -init 1000000 -step 1000000 -max 10000000 -type float -dist angular  -ml 0.36 -m 32 -efc 75 -alpha 0.85 -b 2 -f 0 -in ${ds[$name]} -q ${query[$name]} -g ${gt[$name]} -efs 10,20,50,100,200,300,400,500 -k 10 -w 0 -le 1 -gtf ${gtf[$name]} > $(protect results/0715-beamSearch2/rebuild/occlude.log)
#./dt_rprune -init 1000000 -step 1000000 -max 10000000 -type float -dist angular  -ml 0.36 -m 32 -efc 75 -alpha 0.85 -b 2 -f 0 -in ${ds[$name]} -q ${query[$name]} -g ${gt[$name]} -efs 10,20,50,100,200,300,400,500 -k 10 -w 0 -le 1 -gtf ${gtf[$name]} > $(protect results/0715-beamSearch2/rebuild/rprune.log)
#./dt_all -init 1000000 -step 1000000 -max 10000000 -type float -dist angular  -ml 0.36 -m 32 -efc 75 -alpha 0.85 -b 2 -f 0 -in ${ds[$name]} -q ${query[$name]} -g ${gt[$name]} -efs 10,20,50,100,200,300,400,500 -k 10 -w 0 -le 1 -gtf ${gtf[$name]} > $(protect results/0715-beamSearch2/rebuild/all.log)

name="deep10m"
./test_rebuild -init 1000000 -step 1000000 -max 10000000 -type float -dist angular  -ml 0.36 -m 32 -efc 75 -alpha 0.85 -b 2 -f 0 -in ${ds[$name]} -q ${query[$name]} -g ${gt[$name]} -efs 10,20,50,100,200,300,400,500 -k 10 -w 0 -le 1 -gtf ${gtf[$name]} > $(protect results/0715-beamSearch2/rebuild/deep-10M.log)
name="deep100m"
./test_rebuild -init 10000000 -step 10000000 -max 100000000 -type float -dist angular  -ml 0.36 -m 32 -efc 75 -alpha 0.85 -b 2 -f 0 -in ${ds[$name]} -q ${query[$name]} -g ${gt[$name]} -efs 10,20,50,100,200,300,400,500 -k 10 -w 0 -le 1 -gtf ${gtf[$name]}/100M/ > $(protect results/0715-beamSearch2/rebuild/deep-100M.log)

name="openai5m"
./test_rebuild -init 500000 -step 500000 -max 5000000 -type float -dist angular  -ml 0.36 -m 32 -efc 75 -alpha 0.85 -b 2 -f 0 -in ${ds[$name]} -q ${query[$name]} -g ${gt[$name]} -efs 10,20,50,100,200,300,400,500 -k 10 -w 0 -le 1 -gtf ${gtf[$name]} > $(protect results/0715-beamSearch2/rebuild/openai-5M-500K.log)
