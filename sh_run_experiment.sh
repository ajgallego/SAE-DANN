#!/bin/bash

#python -u py_dasae.py -path datasets -db1 sal -db2 dibco2016 -s 128 -l 5 -f 128 -gpu 0
#python -u py_dasae.py -path datasets -db1 dibco2016 -db2 palm0
#exit

gpu=0

source=sal            				# 'dibco2016','dibco2014','palm0','palm1','phi','ein','sal','voy','bdi','all'
target=dibco2016     		#
window=256
step=120
layers=5
filters=128
kernel=5
e=300
b=12   									# 64 128 256
page=-1
lda=0.001
#lr=0.5				  # 0.5  1.0
options=		#--truncate

python -u py_dasae.py -path datasets -db1 ${source} -db2 ${target} \
				-w ${window} -s ${step} \
				-l ${layers} -f ${filters} -k ${kernel} \
				-lda ${lda} \
				-e ${e} -b ${b} -page ${page} \
				-gpu ${gpu} \
				${options}
				> out_DANN_${source}-${target}_w${window}_s${step}_l${layers}_f${filters}_k${kernel}_lda${lda}_e${e}_b${b}_page${page}.txt


exit


cat out.txt | grep 'Result:'
cat out.txt | grep '* New selected samples'
cat out.txt | grep ' - Source test set'


# GRID SEARCH

for b in 256 512; do   # 16 32 64 128 256 512
	for lda in 0.5 1.0 1.5 2.0; do
	    	for lr1 in 0.5 1.0 1.5; do
			for lr2 in 0.5 1.0 1.5; do

				python -u py_dann.py -db ${db} -mode cnn -e ${e} -b ${b} -fold -1 -lda ${lda} -lr1 ${lr1} -lr2 ${lr2} -gpu $gpu >  OUT/out_${db}_CNN_e${e}_b${b}_lda${lda}_lr1_${lr1}_lr2_${lr2}.txt

				python -u py_dann.py -db ${db} -mode dann -e ${e} -b ${b} -fold -1 -lda ${lda} -lr1 ${lr1} -lr2 ${lr2} -gpu $gpu >  OUT/out_${db}_DANN_e${e}_b${b}_lda${lda}_lr1_${lr1}_lr2_${lr2}.txt

			done
		done
	done
done

