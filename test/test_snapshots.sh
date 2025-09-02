#!/usr/bin/env bash

set -x

logdir=results/0413-chrnlist/m32efc75/0-20K-1M-BIGANN/
mkdir -p $logdir
./test_snapshots -init 20000 -step 20000 -max 1000000 -type uint8 -dist L2 -in ./ANNdataset/BIGANN/base.1B.u8bin:u8bin -ml 0.36 -m 32 -efc 75 -alpha 0.82 -b 2 -snapshot 0 -graph simple &> ${logdir}/simple-disable.log
./test_snapshots -init 20000 -step 20000 -max 1000000 -type uint8 -dist L2 -in ./ANNdataset/BIGANN/base.1B.u8bin:u8bin -ml 0.36 -m 32 -efc 75 -alpha 0.82 -b 2 -snapshot 1 -graph simple &> ${logdir}/simple-enable.log
#./test_snapshots -init 20000 -step 20000 -max 1000000 -type uint8 -dist L2 -in ./ANNdataset/BIGANN/base.1B.u8bin:u8bin -ml 0.36 -m 32 -efc 75 -alpha 0.82 -b 2 -snapshot 0 -graph pam &> ${logdir}/pam-disable.log
#./test_snapshots -init 20000 -step 20000 -max 1000000 -type uint8 -dist L2 -in ./ANNdataset/BIGANN/base.1B.u8bin:u8bin -ml 0.36 -m 32 -efc 75 -alpha 0.82 -b 2 -snapshot 1 -graph pam &> ${logdir}/pam-enable.log

logdir=results/0413-chrnlist/m32efc75/0-200K-10M-BIGANN/
mkdir -p $logdir
./test_snapshots -init 200000 -step 200000 -max 10000000 -type uint8 -dist L2 -in ./ANNdataset/BIGANN/base.1B.u8bin:u8bin -ml 0.36 -m 32 -efc 75 -alpha 0.82 -b 2 -snapshot 0 -graph simple &> ${logdir}/simple-disable.log
./test_snapshots -init 200000 -step 200000 -max 10000000 -type uint8 -dist L2 -in ./ANNdataset/BIGANN/base.1B.u8bin:u8bin -ml 0.36 -m 32 -efc 75 -alpha 0.82 -b 2 -snapshot 1 -graph simple &> ${logdir}/simple-enable.log
#./test_snapshots -init 200000 -step 200000 -max 10000000 -type uint8 -dist L2 -in ./ANNdataset/BIGANN/base.1B.u8bin:u8bin -ml 0.36 -m 32 -efc 75 -alpha 0.82 -b 2 -snapshot 0 -graph pam &> ${logdir}/pam-disable.log
#./test_snapshots -init 200000 -step 200000 -max 10000000 -type uint8 -dist L2 -in ./ANNdataset/BIGANN/base.1B.u8bin:u8bin -ml 0.36 -m 32 -efc 75 -alpha 0.82 -b 2 -snapshot 1 -graph pam &> ${logdir}/pam-enable.log

logdir=results/0413-chrnlist/m48efc100/0-20K-1M-BIGANN/
mkdir -p $logdir
./test_snapshots -init 20000 -step 20000 -max 1000000 -type uint8 -dist L2 -in ./ANNdataset/BIGANN/base.1B.u8bin:u8bin -ml 0.36 -m 48 -efc 100 -alpha 0.82 -b 2 -snapshot 0 -graph simple &> ${logdir}/simple-disable.log
./test_snapshots -init 20000 -step 20000 -max 1000000 -type uint8 -dist L2 -in ./ANNdataset/BIGANN/base.1B.u8bin:u8bin -ml 0.36 -m 48 -efc 100 -alpha 0.82 -b 2 -snapshot 1 -graph simple &> ${logdir}/simple-enable.log
#./test_snapshots -init 20000 -step 20000 -max 1000000 -type uint8 -dist L2 -in ./ANNdataset/BIGANN/base.1B.u8bin:u8bin -ml 0.36 -m 48 -efc 100 -alpha 0.82 -b 2 -snapshot 0 -graph pam &> ${logdir}/pam-disable.log
#./test_snapshots -init 20000 -step 20000 -max 1000000 -type uint8 -dist L2 -in ./ANNdataset/BIGANN/base.1B.u8bin:u8bin -ml 0.36 -m 48 -efc 100 -alpha 0.82 -b 2 -snapshot 1 -graph pam &> ${logdir}/pam-enable.log

logdir=results/0413-chrnlist/m48efc100/0-200K-10M-BIGANN/
mkdir -p $logdir
./test_snapshots -init 200000 -step 200000 -max 10000000 -type uint8 -dist L2 -in ./ANNdataset/BIGANN/base.1B.u8bin:u8bin -ml 0.36 -m 48 -efc 100 -alpha 0.82 -b 2 -snapshot 0 -graph simple &> ${logdir}/simple-disable.log
./test_snapshots -init 200000 -step 200000 -max 10000000 -type uint8 -dist L2 -in ./ANNdataset/BIGANN/base.1B.u8bin:u8bin -ml 0.36 -m 48 -efc 100 -alpha 0.82 -b 2 -snapshot 1 -graph simple &> ${logdir}/simple-enable.log
#./test_snapshots -init 200000 -step 200000 -max 10000000 -type uint8 -dist L2 -in ./ANNdataset/BIGANN/base.1B.u8bin:u8bin -ml 0.36 -m 48 -efc 100 -alpha 0.82 -b 2 -snapshot 0 -graph pam &> ${logdir}/pam-disable.log
#./test_snapshots -init 200000 -step 200000 -max 10000000 -type uint8 -dist L2 -in ./ANNdataset/BIGANN/base.1B.u8bin:u8bin -ml 0.36 -m 48 -efc 100 -alpha 0.82 -b 2 -snapshot 1 -graph pam &> ${logdir}/pam-enable.log
