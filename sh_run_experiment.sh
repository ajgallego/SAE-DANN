#!/bin/bash

python -u py_dasae.py -path datasets -db1 dibco2016 -db2 palm0

exit


gpu=0

model=1              # 1 2 3
db=mnist            # mnist  signs  mensural officehome
select=None     # 'mnist', 'mnist_m', 'svhn', 'syn_numbers'   |||   'gtsrb', 'syn_signs'
norm=255            # 255 mean standard
e=300
#ei=25
b=128   # 64 128 256
#lda=0.5
#lr1=0.5				  # 0.5  1.0
#lr2=1.0
#iopt=prob  			# prob', 'diff', 'knn', 'mode'

# -iopt ${iopt} -ei ${ei} -lda ${lda} -lr1 ${lr1} -lr2 ${lr2}

python -u py_dann_incr.py -type dann  -model ${model} -db ${db} -select ${select} \
				-norm ${norm} -e ${e} -b ${b} -fold -1 -gpu $gpu \
                > out_DANN2_model_${model}_${db}_select_${select}_norm_${norm}_e${e}_b${b}.txt

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

