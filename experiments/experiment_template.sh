# ---------------------------------------------------------
# evaluate the concatenation of key of the different heads: 
# ---------------------------------------------------------



# ---------------- with "--concat_heads True" --------------------------

cd ..

# object discovery

python3 main_mvclustering.py --dataset VOC07 --set trainval --rho 0.0 --which_features k --kernel linear  --assign_func uncoupled --normalize l2 --tau_kernel 0.2 --binary_graph --concat_heads True
python3 main_mvclustering.py --dataset VOC12 --set trainval --rho 0.0 --which_features k --kernel linear  --assign_func uncoupled --normalize l2 --tau_kernel 0.2 --binary_graph --concat_heads True

cd unsupervised_saliency_detection

# saliency detection (for the bilateral solver use the same parameters as in Tokencut)

python get_saliency.py --dataset ECSSD --out-dir ECSSD --sigma-spatial 16 --sigma-luma 16 --sigma-chroma 8 --rho 0.0 --vit-feat k --kernel linear --assign_func uncoupled --normalize l2 --tau_kernel 0.2 --binary_graph --concat_heads True
python get_saliency.py --dataset DUTS --out-dir DUTS --sigma-spatial 16 --sigma-luma 16 --sigma-chroma 8 --rho 0.0 --vit-feat k --kernel linear --assign_func uncoupled --normalize l2 --tau_kernel 0.2 --binary_graph --concat_heads True



# ---------------- without "--concat_heads True" --------------------------

cd ..

# object discovery

python3 main_mvclustering.py --dataset VOC07 --set trainval --rho 0.0 --which_features k --kernel linear  --assign_func uncoupled --normalize l2 --tau_kernel 0.2 --binary_graph
python3 main_mvclustering.py --dataset VOC12 --set trainval --rho 0.0 --which_features k --kernel linear  --assign_func uncoupled --normalize l2 --tau_kernel 0.2 --binary_graph

cd unsupervised_saliency_detection

# saliency detection (for the bilateral solver use the same parameters as in Tokencut)

python get_saliency.py --dataset ECSSD --out-dir ECSSD --sigma-spatial 16 --sigma-luma 16 --sigma-chroma 8 --rho 0.0 --vit-feat k --kernel linear --assign_func uncoupled --normalize l2 --tau_kernel 0.2 --binary_graph
python get_saliency.py --dataset DUTS --out-dir DUTS --sigma-spatial 16 --sigma-luma 16 --sigma-chroma 8 --rho 0.0 --vit-feat k --kernel linear --assign_func uncoupled --normalize l2 --tau_kernel 0.2 --binary_graph

