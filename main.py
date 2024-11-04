import multiprocessing as mp
import argparse
from datetime import datetime
from mm_detect import ModelContaminationChecker
from mm_detect.configs.config import supported_methods
from mm_detect.utils.utils import seed_everything
from mm_detect.utils.logger import setting_logger


def parse_args():
    parser = argparse.ArgumentParser()
    # General arguments
    parser.add_argument("--caption_key", type=str, default="",
                        help="The caption key of each data instance.")
    parser.add_argument("--image_key", type=str, default="",
                        help="The key to image content of each data instance.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--eval_data_name", type=str, default="",
                        help="Eval dataset name")  
    parser.add_argument("--eval_data_config_name", type=str, default=None,
                        help="Eval dataset config name")  
    parser.add_argument("--eval_set_key", type=str, default="test",
                        help="Eval set key")
    parser.add_argument("--text_key", type=str, default="",
                        help="The key to text content of each data instance.")
    parser.add_argument("--n_eval_data_points", type=int, default=100,
                        help="The number of (val/test) data points to keep for evaluating contamination")
    parser.add_argument("--method", type=str, choices=supported_methods.keys(),
                        help="you must pass a method name within the list supported_methods")
    parser.add_argument("--output_dir", type=str, default="output",
                        help="Output directory for logging if necessary")

    # Method specific-arguments for model contamination detection
    ### Shared across methods
    parser.add_argument("--model_name", type=str, default=None,
                        help="Model name for service based inference.")
    parser.add_argument("--max_output_tokens", type=int, default=128,
                        help="Max number of output tokens")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Temperature when sampling each sample")

    args = parser.parse_args()

    # Setting global logger name
    # current_date = datetime.now().strftime('%Y%m%d_%H%M%S')
    data = args.dataset_name if args.dataset_name != "" else args.eval_data_name
    data = data.replace("/", "_")
    log_file_name = f"{data}_{args.n_eval_data_points}.txt"
    logger = setting_logger(log_file_name, args.output_dir) 

    logger.warning(args)

    return args

def check_args(args):
    assert args.method in supported_methods, f"Error, {args.method} not in supported methods: {list(supported_methods.keys())}"

def main():
    args = parse_args()
    check_args(args)

    seed_everything(args.seed)

    ContaminationChecker = ModelContaminationChecker

    contamination_checker = ContaminationChecker(args)
    contamination_checker.run_contamination(args.method)

if __name__ == '__main__':
    main()
