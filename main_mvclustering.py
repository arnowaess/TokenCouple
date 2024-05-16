import os
import argparse
import random
import pickle
import torch
import datetime
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from PIL import Image
from networks import get_model
from datasets import ImageDataset, Dataset, bbox_iou, VocDataset
from visualizations import visualize_img, visualize_eigvec, visualize_predictions, visualize_predictions_gt
import matplotlib.pyplot as plt
import time
from unsupervised_saliency_detection import object_discovery
from kernels import kernel_factory
from torchvision import transforms
import seaborn as sn
from pylab import savefig
import warnings  # Suppress specific UserWarnings from PyTorch about upsample

warnings.filterwarnings('ignore', message='Default upsampling behavior when mode=bicubic is changed')
warnings.filterwarnings('ignore', message='The default behavior for interpolate/upsample with float scale_factor changed')

os.environ['OPENBLAS_NUM_THREADS'] = '1'

def chosen_features(from_backbone, args):
    qkv = from_backbone[:, 0, :, 1:, :]  # shape = [3,6,N,64]
    conc_qkv = torch.stack([f.transpose(0, 1).reshape(1, qkv.shape[2], -1) for f in qkv])  # shape = [3,1,N,384]

    if args.concat_everything:  # feature shape: [1,N,1152]
        return conc_qkv[:, 0, :, :].transpose(0, 1).reshape(1, qkv.shape[2], -1).cpu()

    if args.concat_heads:  # feature shape: [1,N,384] or [3,N,384]
        feature_options = {"k": conc_qkv[1], "q": conc_qkv[0], "v": conc_qkv[2],"qkv": conc_qkv[:,0,:,:]}
    else:
        if args.specific_heads:  # feature shape: [#chosen heads,N,64] or [3*#chosen heads,N,64]
            feature_options = {"q": qkv[0,args.specific_heads,:,:], "k": qkv[1,args.specific_heads,:,:], "v": qkv[2,args.specific_heads,:,:],
                               "qkv": torch.cat([qkv[0,args.specific_heads,:,:], qkv[1,args.specific_heads,:,:], qkv[2,args.specific_heads,:,:]], dim=0)}
        else:  # feature shape: [6,N,64] or [18,N,64]
            feature_options = {"q": qkv[0], "k": qkv[1], "v": qkv[2], "qkv": torch.cat([qkv[0], qkv[1], qkv[2]], dim=0)}

    return feature_options[args.which_features].cpu()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Conduct MV-Clustering on Pre-trained Transformer features")

    parser.add_argument("--arch", default="vit_small", type=str, choices=["vit_tiny", "vit_small", "vit_base", "moco_vit_small", "moco_vit_base", "mae_vit_base"], help="Model architecture.")
    parser.add_argument("--patch_size", default=16, type=int, help="Patch resolution of the model.")

    # Use a dataset for object detection
    parser.add_argument("--dataset", default="VOC07", type=str, choices=[None, "VOC07", "VOC12"], help="Dataset name.")
    parser.add_argument("--save-feat-dir", type=str, default=None, help="if save-feat-dir is not None, only computing features and save it into save-feat-dir")
    parser.add_argument("--set", default="train", type=str, choices=["val", "train", "trainval", "test"], help="Path of the image to load.")
    # Or use a single image
    parser.add_argument("--image_path", type=str, default=None, help="If want to apply only on one image, give file path.")

    # Folder used to output visualizations and 
    parser.add_argument("--output_dir", type=str, default="outputs", help="Output directory to store predictions and visualizations.")

    # Evaluation setup
    parser.add_argument("--no_hard", action="store_true", help="Only used in the case of the VOC_all setup (see the paper).")
    parser.add_argument("--no_evaluation", action="store_true", help="Compute the evaluation.")
    parser.add_argument("--save_predictions", default=True, type=bool, help="Save predicted bouding boxes.")

    # Visualization
    parser.add_argument("--visualize", type=str, choices=["attn", "pred", "all", None], default=None, help="Select the different type of visualizations.")

    # TokenCut parameters
    parser.add_argument("--which_features", type=str, default="k", choices=["qkv", "k", "q", "v"], help="Which features to use")
    parser.add_argument("--max_size", type=int, default=None, help="resize image according to the longer edge.")

    # Multi-view RKM
    parser.add_argument("--k_clusters", type=int, default=2, help="How many clusters for segmentation.")
    parser.add_argument("--which_eigen", type=int, default=-1, help="which eigen to be visualized, and use all the eigenvectors if it is -1.")
    parser.add_argument("--kernel", type=str, choices=["rbf", "normpoly", "poly", "linear", "laplacian"], default="rbf", help="Select the type of kernel to use for multi-view RKM.")
    parser.add_argument("--assign_func", type=str, choices=["uncoupled", "mean"], default="mean", help="Use uncoupled-view ensemble or mean-view ensemble of the final clusters prediction.")
    parser.add_argument("--sigma2", type=float, default=50.0, help="parameter for RBF kernel.")
    parser.add_argument("--eta", type=float, default=1.0, help="regularization coefficient.")
    parser.add_argument("--rho", type=float, default=0.0, help="to balance the RKM loss (1) and Tensor-based multi-view loss (rho)")
    parser.add_argument("--kappa", type=float, nargs='+', default=None, help="regularization weight coefficient for each view, weights are provided with spaced inbetween and should add up to 1.")

    # not imortant
    parser.add_argument("--isRot", action="store_true", help="whether use rotation when loading images so as to double the possible views")
    parser.add_argument("--tau_box", type=float, default=None, help="Tau for seperating the kernel matrix.")

    # Use dino-seg proposed method
    parser.add_argument("--dinoseg", action="store_true", help="Apply DINO-seg baseline.")
    parser.add_argument("--dinoseg_head", type=int, default=4)
    parser.add_argument('--resumePth', type=str, help='resume path')

    parser.add_argument("--normalize", type=str, choices=["l1", "l2", "None"], default="None", help="Select the different type of input normalization.")
    parser.add_argument("--binary_graph", action="store_true", help="Generate a binary graph where edge of the Graph will binary. Or using similarity score as edge weight.")
    parser.add_argument("--tau_kernel", nargs='+', type=float, default=None, help="Tau for seperating the kernel matrix.")

    parser.add_argument("--concat_heads", default=False, type=bool, help="Concatenate the patch embeddings of the different heads.")
    parser.add_argument("--concat_everything",default=False, type=bool, help="Concatenate k,q,v into in view [1, #tokens, 64*18].")
    parser.add_argument("--specific_heads", nargs='+', default=False, type=int,
                        help="give indices of one or more specific heads to use, in range 0 to 5.")

    args = parser.parse_args()
    #print(args)

    if args.image_path is not None:
        args.save_predictions = False
        args.no_evaluation = True
        args.dataset = None

    # to visualize all the bounding box
    args.no_evaluation = True

 # -------------------------------------------------------------------------------------------------------
    # Dataset
    # If an image_path is given, apply the method only to the image
    if args.image_path is not None:
        dataset = ImageDataset(args.image_path, args.patch_size, None, args.isRot)
    else:
        if args.dataset == "VOC07":
            root = "datasets/VOC2007"
            year = "2007"
        elif args.dataset == "VOC12":
            root = "datasets/VOC2012"
            year = "2012"
        dataset = VocDataset(args.dataset, root, year, args.set, args.no_hard, args.max_size)

    # -------------------------------------------------------------------------------------------------------
    # Model
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = get_model(args.arch, args.patch_size, device)

    # -------------------------------------------------------------------------------------------------------
    # Directories
    if args.image_path is None:
        args.output_dir = os.path.join(args.output_dir, dataset.name)
    os.makedirs(args.output_dir, exist_ok=True)

    # Naming
    if args.dinoseg:
        # Experiment with the baseline DINO-seg
        if "vit" not in args.arch:
            raise ValueError("DINO-seg can only be applied to tranformer networks.")
        exp_name = f"{args.arch}-{args.patch_size}_dinoseg-head{args.dinoseg_head}"
    else:
        # Experiment with MVClustering
        exp_name = f"MVClustering-{args.arch}"
        
    # set kernel params
    if args.kernel == "rbf":
        kernel_param = {"sigma2": args.sigma2}
        exp_name += f"{args.patch_size}_{args.which_features}_{args.kernel}_sig{args.sigma2}"
    elif args.kernel == "linear":
        kernel_param = {'0':0}
        exp_name += f"{args.patch_size}_{args.which_features}_{args.kernel}"
    elif args.kernel == "laplacian":
        kernel_param = {"sigma2": args.sigma2}
        exp_name += f"{args.patch_size}_{args.which_features}_{args.kernel}_sig{args.sigma2}"
    exp_name += f"_rho{args.rho}_norm{args.normalize}_binary{args.binary_graph}_tauKernel{args.tau_kernel}_tightbox{args.tau_box}_resize{args.max_size}"
    

    nr_of_views_used = 1
    if not args.concat_everything:
        if args.which_features == "qkv": 
            nr_of_views_used *= 3
        if not args.concat_heads:
            if args.specific_heads:
                nr_of_views_used *= len(args.specific_heads)
            else:
                nr_of_views_used *= 6
    print(f"Running Multi-view Clustering with {nr_of_views_used} view(s)")
    
    # assert that if view weights are provided, they match the number of views
    if args.kappa is not None:
        assert len(args.kappa) == nr_of_views_used, f"Length of kappa ({len(args.kappa)}) does not match number of views ({nr_of_views_used})"
    else: 
        # if no view weights have been provided, equally balance the weights
        args.kappa = [1.0/float(nr_of_views_used) for view in range(nr_of_views_used)]

    # if only 1 tau is given for more than 1 view, use the same tau for all views
    if args.tau_kernel is not None:
        if len(args.tau_kernel) == 1:
            args.tau_kernel = [args.tau_kernel[0] for view in range(nr_of_views_used)]

    # Visualization 
    if args.visualize:
        vis_folder = f"{args.output_dir}"
        os.makedirs(vis_folder, exist_ok=True)
        
    if args.save_feat_dir is not None : 
        os.mkdir(args.save_feat_dir)

    # -------------------------------------------------------------------------------------------------------
    # Loop over images
    preds_dict = {}
    cnt = 0

    if args.image_path is not None:
        corloc = np.zeros(1)
        pbar = tqdm(dataset.dataloader)
    else:
        corloc = np.zeros(len(dataset))
        pbar = tqdm(dataset)
    
    start_time = time.time() 
    for im_id, inp in enumerate(pbar):

        # ------------ IMAGE PROCESSING -------------------------------------------
        img = inp[0]

        # Get the name of the image
        im_name = dataset.get_image_name(inp[1])
        resize_ratio = None

        if im_name is None:
            continue  # Pass in case of no gt boxes in the image
        
        # Padding the image with zeros to fit multiple of patch-size
        size_im = (
                img.shape[0],
                int(np.ceil(img.shape[1] / args.patch_size) * args.patch_size),
                int(np.ceil(img.shape[2] / args.patch_size) * args.patch_size),
        )
        paded = torch.zeros(size_im)
        paded[:, : img.shape[1], : img.shape[2]] = img
        img = paded

        if args.image_path is not None:
            h, w, c = inp[0].shape
            init_image_size = (c, h, w)
        else:
            init_image_size = dataset.get_orig_size(im_name)

        # Move to gpu
        if device == torch.device('cuda'):
            img = img.cuda(non_blocking=True)
        
        # Size for transformers
        w_featmap = img.shape[-2] // args.patch_size
        h_featmap = img.shape[-1] // args.patch_size

        # ------------ GROUND-TRUTH -------------------------------------------
        if args.image_path is None:
            gt_bbxs, gt_cls = dataset.extract_gt(inp[1], im_name)

            if gt_bbxs is not None:
                # Discard images with no gt annotations
                # Happens only in the case of VOC07 and VOC12
                if gt_bbxs.shape[0] == 0 and args.no_hard:
                    continue

        # ------------ EXTRACT FEATURES -------------------------------------------
        with torch.no_grad():

            # ------------ FORWARD PASS -------------------------------------------
            # Store the outputs of qkv layer from the last attention layer
            feat_out = {}
            def hook_fn_forward_qkv(module, input, output):
                feat_out["qkv"] = output
            model._modules["blocks"][-1]._modules["attn"]._modules["qkv"].register_forward_hook(hook_fn_forward_qkv)

            # Forward pass in the model
            attentions = model.get_last_selfattention(img[None, :, :, :])

            # Scaling factor
            scales = [args.patch_size, args.patch_size]

            # Dimensions
            nb_im = attentions.shape[0]  # Batch size
            nh = attentions.shape[1]  # Number of heads
            nb_tokens = attentions.shape[2]  # Number of tokens

            # Extract the qkv features of the last attention layer
            qkv = (
                feat_out["qkv"]
                .reshape(nb_im, nb_tokens, 3, nh, -1 // nh)
                .permute(2, 0, 3, 1, 4))
        

            feats = chosen_features(qkv, args)
        
                    
            if args.save_feat_dir is not None : 
                np.save(os.path.join(args.save_feat_dir, im_name.replace('.jpg', '.npy').replace('.jpeg', '.npy').replace('.png', '.npy')), feats.cpu().numpy())
                continue


        # ------------ Apply MV-Clustering ------------------------------------------- 
        pred, objects, foreground, eigenvector = object_discovery.mvclustering(args, feats, [w_featmap, h_featmap], scales, init_image_size, args.kernel, kernel_param, 2, args.assign_func, args.eta, args.rho, resize_ratio)

        if args.visualize == "pred" and args.no_evaluation:
            image = dataset.load_image(im_name, size_im)     
            visualize_predictions(image, pred, vis_folder, im_name)
        if args.visualize == "attn" and args.no_evaluation:
            visualize_eigvec(eigenvector, vis_folder, im_name, [w_featmap, h_featmap], scales)
        if args.visualize == "all" and args.no_evaluation:
            image = dataset.load_image(im_name)
            visualize_predictions(image, pred, vis_folder, im_name)
            visualize_eigvec(eigenvector, vis_folder, im_name, [w_featmap, h_featmap], scales)
 
        # ------------ Visualizations -------------------------------------------
        # Save the prediction
        preds_dict[im_name] = pred

        # Evaluation
        if args.image_path is not None:
            continue

        # Compare prediction to GT boxes
        ious = bbox_iou(torch.from_numpy(pred), torch.from_numpy(gt_bbxs))
        if torch.any(ious >= 0.5):
            corloc[im_id] = 1
        vis_folder = f"{args.output_dir}/{exp_name}"
        os.makedirs(vis_folder, exist_ok=True)
        image = dataset.load_image(im_name)

        cnt += 1
        if cnt % 50 == 0:
            pbar.set_description(f"Found {int(np.sum(corloc))}/{cnt}")

    # Save predicted bounding boxes
    if args.save_predictions:
        folder = f"{args.output_dir}"
        os.makedirs(folder, exist_ok=True)
        filename = os.path.join(folder, "preds.pkl")
        with open(filename, "wb") as f:
            pickle.dump(preds_dict, f)

    # Evaluate
    if args.image_path is None:
        end_time = time.time()
        print(f"corloc: {100*np.sum(corloc)/cnt:.2f} ({int(np.sum(corloc))}/{cnt}), Time cost: {str(datetime.timedelta(milliseconds=int((end_time - start_time)*1000)))} ")
        result_file = os.path.join(folder, 'results.txt')
        with open(result_file, 'w') as f:
            f.write('corloc,%.1f,,\n'%(100*np.sum(corloc)/cnt))
