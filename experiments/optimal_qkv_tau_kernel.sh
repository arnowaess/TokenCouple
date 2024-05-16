# -------------------------------------------------------
# find the optimal tau threshold for q,k and v
# experiment with binarizing the omega's and
# different normalizations
# ---------------------------------------------------------


star=0.1
end=2
increment=2


kernel=rbf
feature=k
norm=l2
tau=0.15

cd ..
#cd unsupervised_saliency_detection



for sigma2 in $(seq $star $increment $end)
	do
	echo "------------------------------------------------------------feature = $feature, tau = $tau, norm = $norm, kernel = $kernel, sigma2 = $sigma2  -----------------------------------------------------------------------------"
	
#	cd ..
	
	python3 main_mvclustering.py --dataset VOC07 --set trainval --which_features $feature --kernel $kernel --sigma2 $sigma2 --normalize $norm --concat_heads True --tau_kernel $tau --binary_graph
#	python3 main_mvclustering.py --dataset VOC12 --set trainval --which_features $feature --kernel $kernel --sigma2 $sigma2 --normalize $norm --concat_heads True --tau_kernel $tau --binary_graph

#	cd unsupervised_saliency_detection

#	python get_saliency.py --dataset ECSSD --out-dir ECSSD --sigma-spatial 16 --sigma-luma 16 --sigma-chroma 8 --vit-feat $feature --kernel $kernel --sigma2 $sigma2 --normalize $norm --concat_heads True --tau_kernel $tau --binary_graph
#	python get_saliency.py --dataset DUTS --out-dir DUTS --sigma-spatial 16 --sigma-luma 16 --sigma-chroma 8 --vit-feat $feature --kernel $kernel --sigma2 $sigma2 --normalize $norm --concat_heads True --tau_kernel $tau --binary_graph

done



