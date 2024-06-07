#!/bin/bash

#python -u py_dasae.py -path datasets -db1 sal -db2 dibco2016 -s 128 -l 5 -f 128 -gpu 0
#python -u py_dasae.py -path datasets -db1 dibco2016 -db2 palm0
#exit

gpu=0
type=cnn			# dann cnn
source="sal"			# 'dibco2016','dibco2014','palm0','palm1','phi','ein','sal','voy','bdi','all'
target="dibco2016"  
window=256
step=120				#	-1  120
layers=6
filters=64			# 64 128
kernel=3			# 3 5
drop=0.2			# 0
e=300					# 300
b=12   				# 64 128 256
page=-1
#lr=0.5				# 0.5  1.0
grl_pos=0
domain_model=2
options=""			#--test  --truncate --save

options_serial=${options// /.}
options_serial=${options_serial////-}

#"sal" "dibco2016" "dibco2014" "palm0" "palm1" "phi" "ein"


for lda in 0.1; do
    for source in "dibco2016" ; do #"sal" "ein" "dibco2014" "dibco2016" "palm0" "palm1" "phi"
        for target in "sal"; do #"sal" "ein" "dibco2014" "dibco2016" "palm0" "palm1" "phi"
            for lda_inc in 0.01; do
                #target=${source}

                #if [ $source == $target ]; then
                #    continue
                #fi

                output_file="out_${type}_${source}-${target}_w${window}_s${step}_l${layers}_f${filters}_k${kernel}_drop${drop}_lda${lda}_ldainc${lda_inc}_e${e}_b${b}_page${page}${options_serial}_grlpos${grl_pos}_dmodel${domain_model}.txt"
                echo $output_file

		    	python -u py_dasae.py -type ${type} \
								        -path datasets -db1 ${source} -db2 ${target} \
								        -w ${window} -s ${step} \
								        -l ${layers} -f ${filters} -k ${kernel} -drop ${drop} \
								        -lda ${lda} \
				                        -lda_inc ${lda_inc} \
								        -e ${e} -b ${b} -page ${page} \
								        -gpu ${gpu} \
                                        -gpos ${grl_pos} \
                                        -d_model ${domain_model} \
								        ${options} \
								        > ${output_file}
            done        
        done
    done
done


