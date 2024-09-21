import argparse

def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="facebook/opt-350m")
    parser.add_argument("--type", type=str, required=True, default="ocr")
    parser.add_argument("--box", type=str, default= '')
    parser.add_argument("--color", type=str, default= '')
    parser.add_argument("--render", action='store_true')
    args = parser.parse_args()
