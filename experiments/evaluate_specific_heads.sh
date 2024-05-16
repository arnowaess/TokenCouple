





feature=k
assign=mean
#tau="0.75 0.75 0.75 0.75 0.75 0.75 0.15 0.15 0.15 0.15 0.15 0.15 0.2 0.2 0.2 0.2 0.2 0.2"
tau="0.15"
#tau="0.15 0.2 0.75"
#tau="0.2 0.75 0.15"
heads="0"
rho=1
echo "----------------------------------------------------------- rho $rho, feature = $feature, tau = $tau, assign = $assign, heads: $heads -----------------------------------------------------------------------------"
cd ..
#python3 main_mvclustering.py --dataset VOC07 --set trainval --which_features $feature --kernel linear --normalize l2 --tau_kernel $tau --binary_graph --rho $rho --assign_func $assign #--concat_heads True
#python3 main_mvclustering.py --concat_heads True --dataset VOC12 --set trainval --which_features $feature --kernel linear --normalize l2 --tau_kernel $tau --binary_graph --rho $rho --assign_func $assign

cd unsupervised_saliency_detection

python get_saliency.py --dataset ECSSD --out-dir ECSSD --sigma-spatial 16 --sigma-luma 16 --sigma-chroma 8  --which_features $feature --kernel linear --normalize l2 --tau_kernel $tau --binary_graph --rho $rho --assign_func $assign --specific_heads $heads #--concat_heads True #--concat_everything True
#python get_saliency.py --concat_heads True --dataset DUTS --out-dir DUTS --sigma-spatial 16 --sigma-luma 16 --sigma-chroma 8  --which_features $feature --kernel linear --normalize l2 --tau_kernel $tau --binary_graph --rho $rho --assign_func $assign


heads="1"
python get_saliency.py --dataset ECSSD --out-dir ECSSD --sigma-spatial 16 --sigma-luma 16 --sigma-chroma 8  --which_features $feature --kernel linear --normalize l2 --tau_kernel $tau --binary_graph --rho $rho --assign_func $assign --specific_heads $heads

heads="2"
python get_saliency.py --dataset ECSSD --out-dir ECSSD --sigma-spatial 16 --sigma-luma 16 --sigma-chroma 8  --which_features $feature --kernel linear --normalize l2 --tau_kernel $tau --binary_graph --rho $rho --assign_func $assign --specific_heads $heads

heads="3"
python get_saliency.py --dataset ECSSD --out-dir ECSSD --sigma-spatial 16 --sigma-luma 16 --sigma-chroma 8  --which_features $feature --kernel linear --normalize l2 --tau_kernel $tau --binary_graph --rho $rho --assign_func $assign --specific_heads $heads

heads="4"
python get_saliency.py --dataset ECSSD --out-dir ECSSD --sigma-spatial 16 --sigma-luma 16 --sigma-chroma 8  --which_features $feature --kernel linear --normalize l2 --tau_kernel $tau --binary_graph --rho $rho --assign_func $assign --specific_heads $heads

heads="5"
python get_saliency.py --dataset ECSSD --out-dir ECSSD --sigma-spatial 16 --sigma-luma 16 --sigma-chroma 8  --which_features $feature --kernel linear --normalize l2 --tau_kernel $tau --binary_graph --rho $rho --assign_func $assign --specific_heads $heads

