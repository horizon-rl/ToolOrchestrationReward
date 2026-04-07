# -*- coding: utf-8 -*-
import json
import random
import argparse
import os
import sys
import logging
import datetime
import multiprocessing
from multiprocessing import Pool, Manager
from functools import partial
from tqdm import tqdm
from utils.logger import Logger
from utils.utils import *

# Load OpenAI API key if available (only needed for response evaluation)
openai_api_fn = "openai_api.txt"
if os.path.exists(openai_api_fn):
    with open(openai_api_fn) as f:
        api_key = f.read().strip()
    os.environ['OPENAI_API_KEY'] = api_key

# from runner.general_runner import ToolCallingEvaluator
# from runner.response_runner import RespEvalRunner
from evaluators.tool_use_evaluator import ToolCallingEvaluator
from evaluators.bfcl_evaluator import BFCLToolCallingEvaluator
from evaluators.response_evaluator import RespEvalRunner

# Setup paths: repo root for repo-relative files and BFCL imports
_script_dir = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.abspath(os.path.join(_script_dir, ".."))

# Global cache for BFCL BookingAPI (shared across forked processes)
_preloaded_cache = None
_preloaded_cache_index = None
_booking_api_patched = False

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str, default="logs/test.log")
    parser.add_argument("--input_file", type=str, default="data/ComplexFuncBench.jsonl")
    parser.add_argument("--model_name", type=str, required=True, help="The name of the model to be evaluated.")
    parser.add_argument('--exp_name', type=str, default='full-1000')
    parser.add_argument("--vllm_url", type=str)
    parser.add_argument("--proc_num", type=int, default=1)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--enable-response-eval", action="store_true", help="Enable LLM-as-judge response quality evaluation (disabled by default)")
    parser.add_argument("--use-bfcl", action="store_true", help="Use BFCL deterministic execution instead of semantic matching (faster, consistent with training)")

    args = parser.parse_args()

    # Create log directory for per-sample logs
    os.makedirs(f"result/{args.model_name}/{args.exp_name}/logs/samples", exist_ok=True)

    args.output_dir = f"result/{args.model_name}/{args.exp_name}.jsonl"
    args.log_dir = f"result/{args.model_name}/{args.exp_name}/logs/samples"
    return args


def process_example(data, args):
    log_dir = f"{args.log_dir}/{data['id']}.log"
    logger = Logger(f"evaluation_logger_{data['id']}", log_dir, logging.DEBUG)

    # [0] Initialization. Tested model (target) and evaluation model
    # Use BFCL evaluator if --use-bfcl flag is set (faster, consistent with training)
    if args.use_bfcl:
        model = BFCLToolCallingEvaluator(args=args, logger=logger)
        logger.info("Using BFCL deterministic execution for evaluation")
    else:
        raise NotImplementedError(
            "The legacy semantic-matching evaluator is not included in this open-source release. "
            "Please run evaluation with --use-bfcl."
        )

    resp_eval_model = RespEvalRunner(args=args, logger=logger) if args.enable_response_eval else None

    logger.info(f"Test Example {data['id']}")
    logger.info(f"Query: {data['conversations'][0]['content']}")
    
    turn_count, call_count = 0, 0
    for turn in data['conversations']:
        if turn['role'] == "assistant" and "function_call" in turn:
            turn_count += 1
            call_count += len(turn["function_call"])
            

    # [1] Running trace sampling for a specific item in data for tool calling
    convs, message, turn_id, correct_count = model.run(data)

    # Count number of function calling turns
    real_turn_count = 0
    for turn in convs:
        if turn['role'] == "assistant" and "function_call" in turn:
            real_turn_count += 1

    # Handle API errors - still record the failure for resumability
    if isinstance(message, dict) and message.get("error_type") == "unknown_error":
        logger.error(f"API Error for sample {data['id']}: {message.get('content', 'Unknown error')}")
        result = {
            "id": data['id'],
            "gen_convs": convs,
            "message": message,
            "count_dict": {
                "success_turn_num": 0,
                "total_turn_num": turn_count,
                "correct_call_num": 0,
                "total_call_num": call_count,
                "real_turn_num": 0
            },
            "resp_eval": None,
            "error": True  # Mark as error for later filtering
        }
        with open(args.output_dir, 'a+') as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            f.flush()
        return result


    # [2] Run model based evaluation on generated response

    if args.enable_response_eval and len(convs) > 0 and convs[-1]['role'] == "assistant" and "content" in convs[-1]:
        gen_response = convs[-1]['content']
        resp_eval_result = resp_eval_model.run(data, gen_response)
    else:
        resp_eval_result = None

    # [3] Results logging
    logger.info(f"Message: {message}")
    logger.info(f"Success turn num = {turn_id}")
    logger.info("-" * 100)

    result = {
        "id": data['id'],
        "gen_convs": convs,
        "message": message,
        "count_dict": {
            "success_turn_num": turn_id,
            "total_turn_num": turn_count,
            "correct_call_num": correct_count,
            "total_call_num": call_count,
            "real_turn_num": real_turn_count
        },
        "resp_eval": resp_eval_result
    }

    with open(args.output_dir, 'a+') as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")
        f.flush()

    return result


# Global wrapper for multiprocessing (must be at module level for pickling)
_global_args = None


def _patch_booking_api_init():
    """
    Monkey-patch BookingAPI.__init__ to skip cache loading.
    Uses pre-loaded cache from main process (inherited via fork).
    """
    global _booking_api_patched, _preloaded_cache, _preloaded_cache_index

    if _booking_api_patched:
        return

    bfcl_root = os.path.join(_repo_root, "BFCL")
    if bfcl_root not in sys.path:
        sys.path.insert(0, bfcl_root)
    from bfcl_eval.eval_checker.multi_turn_eval.func_source_code.booking_api import BookingAPI

    _original_init = BookingAPI.__init__

    def _patched_init(self, use_live_cache: bool = False, auto_save_cache: bool = True):
        """Patched init that uses pre-loaded cache instead of loading from disk."""
        self._api_description = "This tool belongs to the BookingAPI, which provides access to Booking.com travel services."
        self._auto_save_cache = auto_save_cache
        self._cache_file = os.path.join(_repo_root, "environment", "booking_api_cache.json")

        # Use pre-loaded cache (inherited from main process via fork)
        if _preloaded_cache is not None:
            self._cache = _preloaded_cache
            self._cache_index = _preloaded_cache_index
        else:
            # Fallback to original loading if no pre-loaded cache
            self._cache = {}
            self._cache_index = {}
            if os.path.exists(self._cache_file):
                with open(self._cache_file, 'r') as f:
                    self._cache = json.load(f)
                self._build_cache_index()

        # No live API for evaluation
        self._api_client = None

    BookingAPI.__init__ = _patched_init
    _booking_api_patched = True


def _init_worker(args):
    """Initialize worker process with args"""
    global _global_args
    _global_args = args

    # Apply monkey-patch for BFCL mode (uses inherited cache from fork)
    if args.use_bfcl:
        _patch_booking_api_init()


def _process_wrapper(data):
    """Wrapper function for multiprocessing - must be at module level"""
    return process_example(data, _global_args)


def main():
    global _global_args, _preloaded_cache, _preloaded_cache_index
    args = get_args()
    _global_args = args  # Store args globally for multiprocessing

    # Pre-load BFCL cache in main process (will be inherited by forked workers)
    if args.use_bfcl:
        print("Pre-loading BFCL BookingAPI cache (this may take a few minutes)...")
        import time
        bfcl_root = os.path.join(_repo_root, "BFCL")
        if bfcl_root not in sys.path:
            sys.path.insert(0, bfcl_root)
        from bfcl_eval.eval_checker.multi_turn_eval.func_source_code.booking_api import BookingAPI

        start_time = time.time()
        _temp_api = BookingAPI()
        _preloaded_cache = _temp_api._cache
        _preloaded_cache_index = _temp_api._cache_index
        load_time = time.time() - start_time

        total_entries = sum(len(v) for v in _preloaded_cache.values())
        print(f"Cache loaded in {load_time:.1f}s: {len(_preloaded_cache)} functions, {total_entries:,} entries")

        # Apply patch in main process too (for sequential mode)
        _patch_booking_api_init()

    test_data = load_json(args.input_file)
    if args.debug:
        test_data = random.sample(test_data, 3)

    if os.path.exists(args.output_dir):
        finished_data = load_json(args.output_dir)
        finised_ids = [d["id"] for d in finished_data]
    else:
        finised_ids = []
    test_data = [d for d in test_data if d['id'] not in finised_ids]

    # Use multiprocessing if proc_num > 1
    if args.proc_num > 1:
        print(f"Using {args.proc_num} parallel processes...")
        from multiprocessing import Pool
        try:
            with Pool(processes=args.proc_num, initializer=_init_worker, initargs=(args,)) as pool:
                # Use imap for progress bar support
                for _ in tqdm(pool.imap(_process_wrapper, test_data), total=len(test_data)):
                    pass
        except Exception as e:
            from litellm.exceptions import AuthenticationError
            if isinstance(e, AuthenticationError) or "AuthenticationError" in str(type(e)) or "AuthenticationError" in str(e):
                print(f"\n{'='*80}\nCRITICAL: API Authentication Failed - {str(e)}\nTerminating evaluation.\n{'='*80}\n")
                raise SystemExit(1)
            raise
    else:
        # Sequential processing
        for data in tqdm(test_data):
            try:
                _process_wrapper(data)
            except Exception as e:
                # Check for critical errors that should halt evaluation
                import traceback
                from litellm.exceptions import AuthenticationError

                # If it's an authentication error, immediately terminate
                if isinstance(e, AuthenticationError) or "AuthenticationError" in str(type(e)):
                    print(f"\n{'='*80}\nCRITICAL: API Authentication Failed - {str(e)}\nTerminating evaluation.\n{'='*80}\n")
                    raise SystemExit(1)

                # Log detailed error information for other exceptions
                error_msg = f"Error processing sample {data.get('id', 'unknown')}: {str(e)}"
                print(f"\n{'='*80}\n{error_msg}\n{'='*80}")
                print(traceback.format_exc())

                # Write error record to output file for resumability
                error_result = {
                    "id": data.get('id', 'unknown'),
                    "gen_convs": [],
                    "message": {"error_type": "exception", "content": str(e), "traceback": traceback.format_exc()},
                    "count_dict": {
                        "success_turn_num": 0,
                        "total_turn_num": 0,
                        "correct_call_num": 0,
                        "total_call_num": 0,
                        "real_turn_num": 0
                    },
                    "resp_eval": None,
                    "error": True,
                    "exception": True
                }
                try:
                    with open(args.output_dir, 'a+') as f:
                        f.write(json.dumps(error_result, ensure_ascii=False) + "\n")
                        f.flush()
                except Exception as write_error:
                    print(f"Failed to write error record: {write_error}")

if __name__ == '__main__':
    # Use 'fork' to inherit pre-loaded cache from main process (copy-on-write)
    # This avoids re-loading the 25GB cache in each worker process
    multiprocessing.set_start_method('fork')
    main()
