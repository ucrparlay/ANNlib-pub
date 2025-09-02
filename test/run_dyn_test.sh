./dyn_test -init 100000 -step 100000 -max 1000000 -type uint8 -dist L2 -in ./ANN_dataset/BIGANN/base.1B.u8bin:u8bin -ml 0.36 -m 32 -efc 128 -alpha 0.82 -b 2 -q ./ANN_dataset/BIGANN/query.public.10K.u8bin:u8bin -ef 100 -k 10 > results/BIGANN/100K-100K-1M-100.log

./dyn_test -init 1000000 -step 100000 -max 10000000 -type uint8 -dist L2 -in ./ANN_dataset/BIGANN/base.1B.u8bin:u8bin -ml 0.36 -m 32 -efc 128 -alpha 0.82 -b 2 -q ./ANN_dataset/BIGANN/query.public.10K.u8bin:u8bin -ef 100 -k 10 > results/BIGANN/1M-100K-10M-100.log

./dyn_test -init 1000000 -step 200000 -max 10000000 -type uint8 -dist L2 -in ./ANN_dataset/BIGANN/base.1B.u8bin:u8bin -ml 0.36 -m 32 -efc 128 -alpha 0.82 -b 2 -q ./ANN_dataset/BIGANN/query.public.10K.u8bin:u8bin -ef 100 -k 10 > results/BIGANN/1M-200K-10M-100.log

./dyn_test -init 1000000 -step 500000 -max 10000000 -type uint8 -dist L2 -in ./ANN_dataset/BIGANN/base.1B.u8bin:u8bin -ml 0.36 -m 32 -efc 128 -alpha 0.82 -b 2 -q ./ANN_dataset/BIGANN/query.public.10K.u8bin:u8bin -ef 100 -k 10 > results/BIGANN/1M-500K-10M-100.log

./dyn_test -init 1000000 -step 1000000 -max 10000000 -type uint8 -dist L2 -in ./ANN_dataset/BIGANN/base.1B.u8bin:u8bin -ml 0.36 -m 32 -efc 128 -alpha 0.82 -b 2 -q ./ANN_dataset/BIGANN/query.public.10K.u8bin:u8bin -ef 100 -k 10 > results/BIGANN/1M-1M-10M-100.log

./dyn_test -init 10000000 -step 10000000 -max 10000000 -type uint8 -dist L2 -in ./ANN_dataset/BIGANN/base.1B.u8bin:u8bin -ml 0.36 -m 32 -efc 128 -alpha 0.82 -b 2 -q ./ANN_dataset/BIGANN/query.public.10K.u8bin:u8bin -ef 100 -k 10 > results/BIGANN/10M-10M-10M-100.log
