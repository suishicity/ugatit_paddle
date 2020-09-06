import os
import glob
import yaml
import cv2
import argparse
import paddle
import paddle.fluid as fluid
from paddle.fluid.dygraph import to_variable

from networks import ResnetGenerator
from utils import denormalize, normalize

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weight_path", type=str, required=True, help="weight path of generator")
    parser.add_argument("--input", type=str, required=True, help="input file path or input diretory of images")
    parser.add_argument("--output", type=str, default="output", help="path to save the result")
    # parser.add_argument("--direction", type=str, default='AtoB', help="translate direction of image domain")
    parser.add_argument("--light", action='store_true', help="whether to use light model")
    parser.add_argument("--use_gpu", type=int, default=1, help="whether to use gpu")

    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)
    return args


def main():
    args = parse_args()
    if os.path.isfile(args.input):
        files = [args.input]
    else:
        files = glob.glob(os.path.join(args.input, '*'))
        files.sort()
    img_size = (256, 256)
    weight_path = args.weight_path

    place = fluid.CUDAPlace() if args.use_gpu else fluid.CPUPlace()
    fluid.dygraph.enable_dygraph()
    net = ResnetGenerator(3, 3, ngf=64, n_blocks=4, img_size=img_size, light=args.light)
    net.load_dict(fluid.load_dygraph(weight_path)[0])
    net.eval()

    for fname in files:
        img = cv2.cvtColor(cv2.imread(fname, 1), cv2.COLOR_BGR2RGB)
        a = normalize(img)
        a = to_variable(a[None, :])
        with fluid.dygraph.no_grad():
            b = net(a)[0]
        b = denormalize(b[0].numpy(), transpose=True, out_format='BGR')
        out_fname = os.path.join(args.output, os.path.basename(fname))
        cv2.imwrite(out_fname, b)

    
if __name__ == "__main__":
    main()