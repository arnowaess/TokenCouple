import sys

import warnings
# Suppress specific UserWarnings from PyTorch related to upsampling
warnings.filterwarnings('ignore', category=UserWarning, message=
'Default upsampling behavior when mode=bicubic is changed')
warnings.filterwarnings('ignore', category=UserWarning, message=
'The default behavior for interpolate/upsample with float scale_factor changed')

sys.path.append('./model')
import dino  # model

import object_discovery
import argparse
import utils
import bilateral_solver
import os

from shutil import copyfile
import PIL.Image as Image
import cv2
import numpy as np
from tqdm import tqdm

from torchvision import transforms
import metric
import matplotlib.pyplot as plt
import skimage
import torch
import torch.nn.functional as F

# Image transformation applied to all images
ToTensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406),
                         (0.229, 0.224, 0.225)), ])


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
    

def get_tokencut_binary_map(img_pth, backbone, patch_size, tau):
    I = Image.open(img_pth).convert('RGB')

    # needed later to rescale the foreground feature map mask back to the original image scale
    w_h_original_image = I.size

    # resize the image height and width to the nearest multiples of 16 
    I_resize, w, h, feat_w, feat_h = utils.resize_pil(I, patch_size)

    tensor = ToTensor(I_resize).unsqueeze(0).cuda()

    # i change the dino.ViTFeat class to output the entire qkv, instead of just the chose q,k,v or or all
    # this way we can choose the number of views just like in object discovery 
    qkv = backbone(tensor)
    # qkv.shape = [3,1,6,N,64], 6 and 64 hold for vit small


    feats = chosen_features(qkv, args)
    #print(feats.shape)


    scales = [args.patch_size, args.patch_size]
    resize_ratio = None
    init_image_size = [3, h, w]

    # switch Ncut for mvclustering
    _, _, mask, _ = (
        object_discovery.mvclustering(args, feats, [feat_h, feat_w], scales, init_image_size,
                                      args.kernel, kernel_param, 2, args.assign_func,
                                      args.eta, args.rho, resize_ratio))
        
    # ADDITIONAL CODE FROM TOKENCUT
    # upsample the feature map mask to fit the original image size using GPU

    mask = torch.from_numpy(mask).to('cuda')
    bipartition = F.interpolate(mask.unsqueeze(0).unsqueeze(0), size=(w_h_original_image[1],w_h_original_image[0]), mode='nearest').squeeze()
    bipartition = bipartition.cpu().numpy()

    return bipartition, _


def mask_color_compose(org, mask, mask_color=[173, 216, 230]):
    mask_fg = mask > 0.5
    rgb = np.copy(org)
    rgb[mask_fg] = (rgb[mask_fg] * 0.3 + np.array(mask_color) * 0.7).astype(np.uint8)

    return Image.fromarray(rgb)


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# input / output dir
parser.add_argument('--out-dir', type=str, help='output directory')
parser.add_argument('--vit-arch', type=str, default='small', choices=['base', 'small'], help='which architecture')

parser.add_argument("--which_features", type=str, default="k", choices=["qkv", "k", "q", "v"], help="Which features to use")
parser.add_argument('--patch-size', type=int, default=16, choices=[16, 8], help='patch size')
parser.add_argument('--tau', type=float, default=0.2, help='Tau for tresholding graph')
# tau not used

parser.add_argument('--sigma-spatial', type=float, default=16, help='sigma spatial in the bilateral solver')
parser.add_argument('--sigma-luma', type=float, default=16, help='sigma luma in the bilateral solver')
parser.add_argument('--sigma-chroma', type=float, default=8, help='sigma chroma in the bilateral solver')

parser.add_argument('--dataset', type=str, default=None, choices=['ECSSD', 'DUTS', None], help='which dataset?')
parser.add_argument('--nb-vis', type=int, default=1, help='nb of visualization')
parser.add_argument('--img-path', type=str, default=None, help='single image visualization')

# MVKSCR arguments:
parser.add_argument("--k_clusters", type=int, default=2, help="How many clusters for segmentation.")
parser.add_argument("--which_eigen", type=int, default=-1,
                    help="which eigen to be visualized, and use all the eigenvectors if it is -1.")
parser.add_argument("--kernel", type=str, choices=["rbf", "normpoly", "poly", "linear", "laplacian"], default="rbf",
                    help="Select the type of kernel to use for multi-view RKM.")
parser.add_argument("--assign_func", type=str, choices=["uncoupled", "mean"], default="mean",
                    help="Use uncoupled-view ensemble or mean-view ensemble of the final clusters prediction.")
parser.add_argument("--sigma2", type=float, default=50.0, help="parameter for RBF kernel.")
parser.add_argument("--eta", type=float, default=1.0, help="regularization coefficient.")
parser.add_argument("--rho", type=float, default=0.0,
                    help="to balance the RKM loss (1) and Tensor-based multi-view loss (rho)")
parser.add_argument("--isRot", action="store_true",
                    help="whether use rotation when loading images so as to double the possible views")
parser.add_argument("--normalize", type=str, choices=["l1", "l2", "None"], default="None",
                    help="Select the different type of input normalization.")
parser.add_argument("--binary_graph", action="store_true",
                    help="Generate a binary graph where edge of the Graph will binary. Or using similarity score "
                         "as edge weight.")


parser.add_argument("--tau_kernel", nargs='+', type=float, default=None, help="Tau for seperating the kernel matrix.")


parser.add_argument("--concat_everything",default=False, type=bool, help="Concatenate k,q,v into in view [1, #tokens, 64*18].")
parser.add_argument("--concat_heads", default=False, type=bool,
                    help="Concatenate the patch embeddings of the different heads.")

parser.add_argument("--specific_heads", nargs='+', default=False, type=int,
                    help="give indices of one or more specific heads to use, in range 0 to 5.")

parser.add_argument("--kappa", type=float, nargs='+', default=None,
                    help="regularization weight coefficient for each view, weights are provided with spaced inbetween and should add up to 1.")

args = parser.parse_args()
#print(args)

# always use small and 16
if args.vit_arch == 'base' and args.patch_size == 16:
    url = "/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth"
    feat_dim = 768
elif args.vit_arch == 'base' and args.patch_size == 8:
    url = "/dino/dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth"
    feat_dim = 768
elif args.vit_arch == 'small' and args.patch_size == 16:
    url = "/dino/dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth"
    feat_dim = 384
elif args.vit_arch == 'base' and args.patch_size == 8:
    url = "/dino/dino_deitsmall8_300ep_pretrain/dino_deitsmall8_300ep_pretrain.pth"

#
backbone = dino.ViTFeat(url, feat_dim, args.vit_arch, args.which_features, args.patch_size)

msg = 'Load {} pre-trained feature...'.format(args.vit_arch)
#print(msg)
backbone.eval()
backbone.cuda()

if args.dataset == 'ECSSD':
    args.img_dir = '../datasets/ECSSD/img'
    args.gt_dir = '../datasets/ECSSD/gt'
elif args.dataset == 'DUTS':
    args.img_dir = '../datasets/DUTS_Test/img'
    args.gt_dir = '../datasets/DUTS_Test/gt'
elif args.dataset is None:
    args.gt_dir = None

# set kernel params
if args.kernel == "rbf":
    kernel_param = {"sigma2": args.sigma2}
elif args.kernel == "linear":
    kernel_param = {'0': 0}
elif args.kernel == "laplacian":
    kernel_param = {"sigma2": args.sigma2}



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

if args.out_dir is not None and not os.path.exists(args.out_dir):
    os.mkdir(args.out_dir)
if args.img_path is not None:
    args.nb_vis = 1
    img_list = [args.img_path]
else:
    img_list = sorted(os.listdir(args.img_dir))

count_vis = 0
mask_lost = []
mask_bfs = []
gt = []
for img_name in tqdm(img_list):
    if args.img_path is not None:
        img_pth = img_name
        img_name = img_name.split("/")[-1]
       # print(img_name)
    else:
        img_pth = os.path.join(args.img_dir, img_name)

    # uses mvclustering instead of tokencut now to get the bipartition
    bipartition, _ = get_tokencut_binary_map(img_pth, backbone, args.patch_size, args.tau)
    mask_lost.append(bipartition)

    # resolve the details of the upscaled binary mask
    output_solver, binary_solver = bilateral_solver.bilateral_solver_output(img_pth, bipartition,
                                                                            sigma_spatial=args.sigma_spatial,
                                                                            sigma_luma=args.sigma_luma,
                                                                            sigma_chroma=args.sigma_chroma)
    mask1 = torch.from_numpy(bipartition).cuda()
    mask2 = torch.from_numpy(binary_solver).cuda()
    if metric.IoU(mask1, mask2) < 0.5:
        binary_solver = binary_solver * -1
    mask_bfs.append(output_solver)

    if args.gt_dir is not None:
        mask_gt = np.array(Image.open(os.path.join(args.gt_dir, img_name.replace('.jpg', '.png'))).convert('L'))
        gt.append(mask_gt)

    if count_vis != args.nb_vis:
        #print(f'args.out_dir: {args.out_dir}, img_name: {img_name}')
        out_name = os.path.join(args.out_dir, img_name)
        out_lost = os.path.join(args.out_dir, img_name.replace('.jpg', '_mask.jpg'))
        out_bfs = os.path.join(args.out_dir, img_name.replace('.jpg', '_BS_mask.jpg'))

        #dont just copy original file
        #copyfile(img_pth, out_name)
        org = np.array(Image.open(img_pth).convert('RGB'))

        # dont save the just upsampled mask
        #mask_color_compose(org, bipartition).save(out_lost)
        mask_color_compose(org, binary_solver).save(out_bfs)
        if args.gt_dir is not None:
            out_gt = os.path.join(args.out_dir, img_name.replace('.jpg', '_gt.jpg'))
            mask_color_compose(org, mask_gt, [255,0,0]).save(out_gt)

        count_vis += 1
    else:
        continue

if args.gt_dir is not None and args.img_path is None:
    #print('Mvclustering evaluation:')
    #print(metric.metrics(mask_lost, gt))
    #print('\n')

    print(f'Mvclustering + bilateral solver evaluation:{metric.metrics(mask_bfs, gt)}')
