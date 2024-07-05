"""
This code is used to convert the pytorch models into an onnx format models.
"""
import argparse
import torch.onnx
from pfld.pfld import PFLDInference
from pfld.mynet import PFLDInference_M


def parse_args():
    parser = argparse.ArgumentParser(description='convert model')
    # general

    # training
    ##  -- optimizer
    parser.add_argument('--input_model', default='', type=str)
    parser.add_argument('--out_model',  default='', type=str)
    parser.add_argument('--network', default="", help="PFLD,PFLDx0.25,mbv1x0.25,shufflenetv2x0.5,mbv3_small", type=str)

    # -- epoch

    args = parser.parse_args()
    return args

def main(args):
    input_img_size = 112  # define input size

    model_path = args.input_model

    checkpoint = torch.load(model_path)
    network_flag =args.network
    if network_flag =='PFLD':
        net = PFLDInference()
    elif network_flag =='PFLDx0.25':
        net = PFLDInference(width_mult=0.25)
    elif network_flag =='mbv1x0.25':
        net = PFLDInference_M(backbone='mobilenetv1-0.25', pretrained=False)
    elif network_flag =='shufflenetv2x0.5':
        net = PFLDInference_M(backbone='shufflenetv2-0.5', pretrained=False)
    elif network_flag =='mobilenetv3_small':
        net = PFLDInference_M(backbone='mobilenetv3_small', pretrained=False)
    net.load_state_dict(checkpoint)
    net.eval()
    net.to("cuda")

    model_name = model_path.split("/")[-1].split(".")[0]
    model_path = args.out_model

    dummy_input = torch.randn(1, 3, 112, 112).to("cuda")

    torch.onnx.export(net, dummy_input, model_path, export_params=True, verbose=False, input_names=['input'],
                      output_names=['pose', 'landms'])

if __name__ == "__main__":
    args = parse_args()
    main(args)
