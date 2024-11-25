import torch
import numpy as np
from network.model_trainer import DiffusionModel
from utils.utils import str2bool, ensure_directory
from utils.utils import num_to_groups
import argparse
import os
from tqdm import tqdm
from PIL import Image
import json as js

def generate_conditional_bdr_json(
    model_path: str,
    output_path: str = "./outputs",
    ema: bool = True,
    num_generate: int = 36,
    start_index: int = 0,
    steps: int = 50,
    truncated_time: float = 0.0,
    json_path="",
    bdr_path="",
    bdr_type=6,
):
    model_name, model_id = model_path.split('/')[-2], model_path.split('/')[-1]
    print(model_name, model_id)
    discrete_diffusion = DiffusionModel.load_from_checkpoint(model_path).cuda()
    postfix = f"{model_name}_{model_id}_{ema}_{steps}_{truncated_time}_conditional"
    root_dir = os.path.join(output_path, postfix)


    # load json file
    with open(json_path,"r") as f:
            data=js.load(f)
    c1=np.array(data["C1"])
    ctensor=np.zeros((3,c1.shape[0]))
    ctensor[0]=c1
    ctensor[1]=np.array(data["C2"])
    ctensor[2]=np.array(data["C3"])

    # load bdr
    mybdr=np.zeros((16,1,128,128))
    for i in range(16):
        # fixed bdr
        ei=bdr_type
        path=os.path.join(bdr_path,str(ei)+".png")
        with open(path, "rb") as f:
            myimg = Image.open(f)
            myimg.load()
        myimg = np.array(myimg.convert("L"))
        myimg = myimg.astype(np.float32) / 127.5 - 1
        bdr=np.ones_like(myimg)
        idx=np.where(myimg[0,:]==-1)[0]
        bdr[idx,:]=-1
        bdr[:,idx]=-1
        mybdr[i]=bdr[np.newaxis,:]

    # print("mybdr: ", mybdr)

    ensure_directory(root_dir)
    batches = num_to_groups(ctensor.shape[1], 16)
    generator = discrete_diffusion.ema_model if ema else discrete_diffusion.model
    index = start_index

    gathered_samples=[]
    j=0
    for batch in batches:
        res_tensor = generator.sample_conditional_bdr_json(
            batch_size=batch, steps=steps, truncated_index=truncated_time,C=ctensor[:,j:j+batch].T,mybdr=mybdr[:batch])
        gathered_samples.extend(post_process(res_tensor).cpu().numpy())
        j+=16
    print(len(gathered_samples),"   ", gathered_samples[0], "   ", root_dir)
    save_as_npz(gathered_samples, root_dir+"/generated_samples.npz")


def post_process(res_tensor):
    res_tensor = ((res_tensor+1)*127.5).clamp(0,255).to(torch.uint8)
    res_tensor = res_tensor.permute(0,2,3,1)
    res_tensor = res_tensor.contiguous()
    return res_tensor

def save_as_npz(gathered_samples,path):
    arr = np.array(gathered_samples)
    np.savez(path,arr)

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(description='generate something')
    parser.add_argument("--generate_method", type=str, default='generate_based_on_bdr_json',
                        # default='generate_conditional_bdr_json',
                        help="please choose :\n \
                            1. 'generate_conditional_bdr_json' \n  \n \ ")

    parser.add_argument("--model_path", type=str, default="C:/Users/Liuxk/PycharmProjects/BDR_diffusion/results/model/epoch=19.ckpt")
    parser.add_argument("--output_path", type=str, default="./outputs")
    parser.add_argument("--ema", type=str2bool, default=True)
    parser.add_argument("--num_generate", type=int, default=16)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--truncated_time", type=float, default=0.0)
    parser.add_argument("--data_class", type=str, default="microstructure")
    parser.add_argument("--text_w", type=float, default=1.0)
    parser.add_argument("--image_path", type=str, default="test.png")
    parser.add_argument("--image_name", type=str2bool, default=False)
    parser.add_argument("--elevation", type=float, default=0.)
    parser.add_argument("--kernel_size", type=float, default=4.)
    parser.add_argument("--verbose", type=str2bool, default=False)
    # json file of elastic tensors
    parser.add_argument("--json_path", type=str, default="C:/Users/Liuxk/PycharmProjects/BDR_diffusion/sample/nsample.json")
    parser.add_argument("--bdr_path", type=str, default="C:/Users/Liuxk/PycharmProjects/BDR_diffusion/bdr")
    parser.add_argument("--bdr_type",type=int,default=2)

    args = parser.parse_args()
    print(args)
    method = (args.generate_method).lower()
    ensure_directory(args.output_path)

    if method == "generate_based_on_bdr_json":
        generate_conditional_bdr_json(model_path=args.model_path, num_generate=args.num_generate,
                               output_path=args.output_path, ema=args.ema, start_index=args.start_index, steps=args.steps,
                               truncated_time=args.truncated_time,json_path=args.json_path,bdr_path=args.bdr_path,bdr_type=args.bdr_type)
    else:
        raise NotImplementedError
