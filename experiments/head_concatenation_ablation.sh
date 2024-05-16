########################################################
# compare multi-view to head concatenation for q,k,v
# we keep the kappas shared equally among views
# we also need to evaluate --assign_function and --rho
########################################################

#--concat_heads True no longer here except when using three views qkv concat

feature=qkv
assign=mean
#tau="0.75 0.75 0.75 0.75 0.75 0.75 0.15 0.15 0.15 0.15 0.15 0.15 0.2 0.2 0.2 0.2 0.2 0.2"
tau="0.75 0.15 0.2"

star=0.6
end=0.6
increment=-0.1
for rho in $(seq $star $increment $end)
do
	echo "----------------------------------------------------------- rho $rho, feature = $feature, tau = $tau, assign = $assign -----------------------------------------------------------------------------"

	cd ..
	#python3 main_mvclustering.py --dataset VOC07 --set trainval --which_features $feature --kernel linear --normalize l2 --tau_kernel $tau --binary_graph --rho $rho --assign_func $assign #--concat_heads True
	#python3 main_mvclustering.py --concat_heads True --dataset VOC12 --set trainval --which_features $feature --kernel linear --normalize l2 --tau_kernel $tau --binary_graph --rho $rho --assign_func $assign 

	cd unsupervised_saliency_detection
	python get_saliency.py --dataset ECSSD --out-dir ECSSD --sigma-spatial 16 --sigma-luma 16 --sigma-chroma 8  --which_features $feature --kernel linear --normalize l2 --tau_kernel $tau --binary_graph --rho $rho --assign_func $assign --concat_heads True #--concat_everything True
	python get_saliency.py --dataset DUTS --out-dir DUTS --sigma-spatial 16 --sigma-luma 16 --sigma-chroma 8  --which_features $feature --kernel linear --normalize l2 --tau_kernel $tau --binary_graph --rho $rho --assign_func $assign --concat_heads True
done


# to evaluate eta:
#for i in 4 3 2 1 0
#do
#	eta=$(bc <<< "scale=10; 10^(-$i)")
#	echo "-------------------------------------------------------------------- features $feature, DONT concat heads, assign $assign, rho 1, eta $eta --------------------------------------------------------------------------"
#        python3 main_mvclustering.py --dataset VOC07 --set trainval --which_features $feature --kernel linear  --normalize l2 --binary_graph --tau_kernel $tau --assign_func $assign --rho 1 --eta $eta
	#python3 main_mvclustering.py --dataset VOC12 --set trainval --which_features $feature --kernel linear  --normalize l2 --binary_graph --tau_kernel $tau --assign_func $assign --rho 1 --eta $eta
#done

