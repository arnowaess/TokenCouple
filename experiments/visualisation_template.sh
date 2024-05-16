
# ----------------------------------------------------------------------------------------------------
# template code to visualize the original image, eigen attention map, bounding box, salient mask and salient mask + BS
# ----------------------------------------------------------------------------------------------------

# change the parameters accordingly

cd ..

python3 main_mvclustering.py --image_path examples/VOC07_000064.jpg --visualize all --output_dir ./outputs --rho 0.0 --which_features k --kernel linear  --assign_func uncoupled --normalize l2 --tau_kernel 0.2 --binary_graph --concat_heads True

cd unsupervised_saliency_detection 

python get_saliency.py --sigma-spatial 16 --sigma-luma 16 --sigma-chroma 8 --img-path ../examples/VOC07_000064.jpg --out-dir ../outputs --vit-feat k --kernel linear --assign_func uncoupled --normalize l2 --binary_graph --tau_kernel 0.2 --concat_heads True



# all images go to "segmentation_rkm-master/outputs/" directory, to copy to local machine, run following FROM LOCAL MACHINE:
#scp -r temse:/users/sista/cheny/arno/segmentation_rkm-master/outputs/* \Users\arnow\Documents\MANAMA\thesis\SSH_transfers\from_remote
