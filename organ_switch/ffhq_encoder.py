from argparse import Namespace
from PIL import Image
import torch
import torchvision.transforms as transforms
from datasets import augmentations
from utils.common import tensor2im, log_input_image
from models.psp import pSp
import dlib
import time
import numpy as np

def run_alignment(image_path):
    from scripts.align_all_parallel import align_face
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    aligned_image = align_face(filepath=image_path, predictor=predictor)
    
#   print("Aligned image has shape: {}".format(aligned_image.size))
    return aligned_image

def run_on_batch(inputs, net, latent_mask=None):
    if latent_mask is None:
        result_batch = net(inputs.to("cuda").float(), randomize_noise=False)
    else:
        result_batch = []
        for image_idx, input_image in enumerate(inputs):
            # get latent vector to inject into our input image
            vec_to_inject = np.random.randn(1, 512).astype('float32')
            _, latent_to_inject = net(torch.from_numpy(vec_to_inject).to("cuda"),
                                      input_code=True,
                                      return_latents=True)
            # get output image with injected style vector
            res = net(input_image.unsqueeze(0).to("cuda").float(),
                      latent_mask=latent_mask,
                      inject_latent=latent_to_inject)
            result_batch.append(res)
        result_batch = torch.cat(result_batch, dim=0)
    return result_batch


def face_reconstruction(image_path, toonify):

    tic = time.time()

    # Load ffhq_encoder model
    if toonify == False:
        model_path = "pretrained_models/psp_ffhq_encode.pt"
    else:
        model_path = "pretrained_models/psp_ffhq_toonify.pt"

    ckpt = torch.load(model_path, map_location='cpu')
    opts = ckpt['opts']
    opts['checkpoint_path'] = model_path
    if 'learn_in_w' not in opts:
        opts['learn_in_w'] = False
    if 'output_size' not in opts:
        opts['output_size'] = 1024
    opts = Namespace(**opts)
    net = pSp(opts)
    net.eval()
    net.cuda()

    toc = time.time()
    # print('Encoder successfully loaded!')
    print('Load ffhq encoder took {:.4f} seconds.'.format(toc - tic))

    # Alignment
    tic = time.time()
    original_image = Image.open(image_path)
    original_image = original_image.convert("RGB")
    input_image = run_alignment(image_path)
    toc = time.time()
    # print('Alignment took {:.4f} seconds.'.format(toc - tic))

    # Image Transformation
    tic = time.time()
    img_transforms = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    transformed_image = img_transforms(input_image)
    toc = time.time()
    # print('Transformation took {:.4f} seconds.'.format(toc - tic))

    # Get output from the model
    latent_mask = None
    with torch.no_grad():
        tic = time.time()
        result_image = run_on_batch(transformed_image.unsqueeze(0), net, latent_mask)[0]
        toc = time.time()
        # print('Inference took {:.4f} seconds.'.format(toc - tic))

    # Visualize the result

    tic = time.time()

    input_vis_image = log_input_image(transformed_image, opts)
    output_image = tensor2im(result_image)
    res = np.concatenate([np.array(input_vis_image.resize((256, 256))),np.array(output_image.resize((256, 256)))], axis=1)
    res_image = Image.fromarray(res)

    toc = time.time()
    # print('Visualization took {:.4f} seconds.'.format(toc - tic))

    return output_image