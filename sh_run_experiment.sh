#!/bin/bash

gpu=0

type=cnn			# dann cnn
source=sal			# 'dibco2016','dibco2014','palm0','palm1','phi','ein','sal','voy','bdi','all'
target=dibco2016     		#
window=256
step=120
layers=5
filters=64			# 64 128
kernel=3			# 3 5
drop=0.2			# 0
e=1				# 300
b=12   				# 64 128 256
page=-1
lda=0.00001			# 0.01  0.001	0.0001	0.00001
#lr=0.5				# 0.5  1.0
options=			#--test  --truncate

python -u py_dasae.py -type ${type} \
				-path datasets -db1 ${source} -db2 ${target} \
				-w ${window} -s ${step} \
				-l ${layers} -f ${filters} -k ${kernel} -drop ${drop} \
				-lda ${lda} \
				-e ${e} -b ${b} -page ${page} \
				-gpu ${gpu} \
				${options} \
				#> out_${type}_${source}-${target}_w${window}_s${step}_l${layers}_f${filters}_k${kernel}_drop${drop}_lda${lda}_e${e}_b${b}_page${page}_${options}.txt

exit

# GRID SEARCH

for layers in 2 3 4 5 6; do   # 16 32 64 128 256 512
	for filters in 8 16 32 64 128 256; do
	    for kernel in 3 5 7; do
			for drop in 0 0.1 0.2 0.5; do
				for lda in 0.1 0.01 0.001 0.0001 0.00001 0.000001; do
					python -u py_dasae.py -type ${type} \
								-path datasets -db1 ${source} -db2 ${target} \
								-w ${window} -s ${step} \
								-l ${layers} -f ${filters} -k ${kernel} -drop ${drop} \
								-lda ${lda} \
								-e ${e} -b ${b} -page ${page} \
								-gpu ${gpu} \
								${options} \
								> out_${type}_${source}-${target}_w${window}_s${step}_l${layers}_f${filters}_k${kernel}_drop${drop}_lda${lda}_e${e}_b${b}_page${page}_${options}.txt
				done
			done
		done
	done
done

