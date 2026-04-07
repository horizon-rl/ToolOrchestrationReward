"""
BFCL-based Tool Calling Evaluator

This evaluator replaces the slow semantic matching (embedding + LLM + API calls)
with BFCL's deterministic execution and reuses the same validation logic as training.

Performance: ~10ms/turn (vs 2-5s/turn with CompareFC)
Consistency: Uses same validation as training (check_ast_validity_graduated)
"""

import sys
import os
import copy
import json
from typing import Dict, List, Tuple, Any, Optional
from evaluators.base_evaluator import BaseModelEvaluator

# Import BFCL execution components (same as training). Optional local verl clone at <repo>/verl.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_verl_repo = os.path.join(_REPO_ROOT, "verl")
if os.path.isdir(os.path.join(_verl_repo, "verl")) and _verl_repo not in sys.path:
    sys.path.insert(0, _verl_repo)
from verl.tools.utils.mcp_clients.BFCLClientManager import BFCLToolClient
from verl.utils.reward_score.complex_tool import safe_get_arguments


class BFCLToolCallingEvaluator(BaseModelEvaluator):
    """
    Tool calling evaluator using BFCL deterministic execution.

    Key differences from ToolCallingEvaluator:
    - No embedding model (no BAAI/bge-large-en-v1.5 loading)
    - No LLM-based comparison (no GPT-4o calls)
    - No real API calls (uses BFCL cache)
    - Same validation logic as training (consistent with RL rewards)
    """

    def __init__(self, args, logger):
        # Don't initialize CompareFC (embedding model) - use BFCL instead
        super().__init__(args, logger, use_compare_fc=False)
        self.model_name = args.model_name
        self.max_completion_tokens = getattr(args, 'max_completion_tokens', 4096)
        self.vllm_url = getattr(args, 'vllm_url', None)

        # Load tool schemas for BFCL (JSONL format — one JSON per line)
        tool_schema_path = os.path.join(
            _REPO_ROOT,
            "environment/booking_api.json",
        )

        tool_schemas = []
        with open(tool_schema_path, 'r') as f:
            for line in f:
                tool_schemas.append(json.loads(line.strip()))

        # Convert to OpenAI function format if needed
        tools = []
        for schema in tool_schemas:
            if 'type' not in schema:
                # Add type field if missing
                tools.append({"type": "function", "function": schema})
            else:
                tools.append(schema)

        # Initialize BFCL client (same as training)
        self.bfcl_client = BFCLToolClient(tools=tools)

        self.logger.info("BFCLToolCallingEvaluator initialized with deterministic BFCL execution")
        self.logger.info(f"Loaded {len(tools)} tool schemas from {tool_schema_path}")

    def get_standard_functions(self, functions):
        """Convert functions to the standard tool format for litellm"""
        return [{"type": "function", "function": copy.deepcopy(func)} for func in functions]

    def get_standard_fc(self, tool_call):
        """Convert tool call to standard function call format"""
        try:
            if hasattr(tool_call, 'function'):
                # litellm format
                return {
                    "name": tool_call.function.name,
                    "arguments": json.loads(tool_call.function.arguments) if isinstance(tool_call.function.arguments, str) else tool_call.function.arguments
                }
            else:
                # fallback for other formats
                return {"name": tool_call['name'], "arguments": tool_call['parameters']}
        except Exception as e:
            self.logger.error(f"Error parsing tool call: {e}")
            return None

    def validate_function_call(
        self,
        pred_call: Dict[str, Any],
        golden_call: Dict[str, Any],
        golden_obs: Any
    ) -> Tuple[bool, Any, Optional[str]]:
        """
        Validate a predicted function call against ground truth.

        Uses BFCL deterministic execution instead of semantic matching.

        Args:
            pred_call: Predicted function call {"name": ..., "arguments": {...}}
            golden_call: Ground truth function call
            golden_obs: Ground truth observation

        Returns:
            (is_valid, observation, error_message)
        """
        # Step 1: Basic format validation
        if pred_call['name'] != golden_call['name']:
            error_msg = f"Function name mismatch: predicted '{pred_call['name']}' but expected '{golden_call['name']}'"
            self.logger.info(error_msg)
            return False, None, error_msg

        # Step 2: Extract arguments safely (handles malformed calls)
        pred_args = safe_get_arguments(pred_call)
        golden_args = safe_get_arguments(golden_call)

        # Step 3: Check required parameters
        missing_params = set(golden_args.keys()) - set(pred_args.keys())
        if missing_params:
            error_msg = f"Missing required parameters: {missing_params}"
            self.logger.info(error_msg)
            return False, None, error_msg

        # Step 4: Execute with BFCL (deterministic, cached)
        try:
            result = self.bfcl_client.call_tool_sync(
                pred_call['name'],
                pred_args
            )

            # Parse result
            if result.isError:
                error_msg = f"BFCL execution error: {result.content[0]['text']}"
                self.logger.warning(error_msg)
                return False, None, error_msg

            # Extract observation
            observation = json.loads(result.content[0]['text'])

            # Step 5: Compare execution result with ground truth observation
            # For BFCL, if the call succeeds and matches the function name/params structure,
            # we consider it valid (deterministic cache ensures consistency)
            is_valid = self._compare_observations(observation, golden_obs)

            if is_valid:
                self.logger.info(f"Function call validated successfully: {pred_call['name']}")
                return True, golden_obs, None  # Return golden obs for consistency
            else:
                error_msg = f"Observation mismatch for {pred_call['name']}"
                self.logger.info(error_msg)
                return False, observation, error_msg

        except Exception as e:
            error_msg = f"Error executing {pred_call['name']}: {str(e)}"
            self.logger.error(error_msg)
            return False, None, error_msg

    def _compare_observations(self, pred_obs: Any, golden_obs: Any) -> bool:
        """
        Compare predicted observation with ground truth.

        For BFCL deterministic execution, we use a lenient comparison:
        - If both have 'status': True, consider them matching
        - If both have 'status': False, consider them matching
        - Otherwise, check for exact match
        """
        # Handle dict observations
        if isinstance(pred_obs, dict) and isinstance(golden_obs, dict):
            # Check status match (most important for booking APIs)
            pred_status = pred_obs.get('status')
            golden_status = golden_obs.get('status')

            if pred_status is not None and golden_status is not None:
                # Both have status - match if same status
                return pred_status == golden_status

            # No status field - try exact match
            return pred_obs == golden_obs

        # Non-dict observations - exact match
        return pred_obs == golden_obs

    def compare_turn_prediction(
        self,
        functions: List[Dict],
        history: List[Dict],
        predictions: List[Dict],
        golden_fcs: List[Dict],
        golden_obs: List[Any]
    ) -> Tuple[List[Dict], Dict[int, Any], List[Dict], Dict[int, Any]]:
        """
        Compare predicted function calls with ground truth.

        This replaces CompareFC.compare_turn_prediction with BFCL execution.

        Returns:
            (error_messages, success_map, success_matched, format_errors)
        """
        error_messages = []
        success_map = {}  # {pred_index: observation}
        success_matched = []  # List of successfully matched golden calls
        format_errors = {}  # {pred_index: error_message}

        # Handle case where all golden calls are already satisfied
        if len(golden_fcs) == 0:
            self.logger.warning("No golden function calls remaining to match")
            return error_messages, success_map, success_matched, format_errors

        # Match predictions to golden calls
        # For now, use simple sequential matching (assumes model calls in order)
        # TODO: Could add Hungarian matching if needed for unordered calls

        matched_golden_indices = set()

        for pred_idx, pred_call in enumerate(predictions):
            # Try to match with remaining golden calls
            matched = False

            for golden_idx, golden_call in enumerate(golden_fcs):
                if golden_idx in matched_golden_indices:
                    continue

                # Try to validate this prediction against this golden call
                is_valid, observation, error_msg = self.validate_function_call(
                    pred_call,
                    golden_call,
                    golden_obs[golden_idx]
                )

                if is_valid:
                    # Match found!
                    success_map[pred_idx] = observation
                    success_matched.append(golden_call)
                    matched_golden_indices.add(golden_idx)
                    matched = True
                    self.logger.info(f"Matched prediction {pred_idx} to golden {golden_idx}")
                    break

            if not matched:
                # No match found for this prediction
                error_messages.append({
                    "error_type": "no_match",
                    "content": f"Could not match prediction {pred_idx}: {pred_call['name']}"
                })
                self.logger.warning(f"No match for prediction {pred_idx}: {pred_call}")

        return error_messages, success_map, success_matched, format_errors

    def run(self, data):
        """
        Run evaluation on a single example.

        This is the main entry point called by evaluation.py
        """
        # Wrap entire method in try-except to catch API errors and convert to dict
        # This prevents pickling errors when passing exceptions between multiprocessing workers
        try:
            return self._run_impl(data)
        except Exception as e:
            # Convert any exception to a dict format that can be pickled
            import traceback
            error_dict = {
                "error_type": "api_error" if "API" in str(type(e).__name__) else "exception",
                "content": str(e),
                "exception_type": type(e).__name__,
                "traceback": traceback.format_exc()
            }
            self.logger.error(f"Exception in run(): {error_dict}")
            # Return empty messages with error
            return [], error_dict, 0, 0

    def _run_impl(self, data):
        """
        Internal implementation of run() - can raise exceptions that will be caught by run()
        """
        # Import here to avoid circular dependency and missing litellm at module load
        from utils.utils import get_llm_response

        convs, functions = data['conversations'], data['functions']

        # Note: BFCL evaluator doesn't use CompareClass, but we need to handle
        # free functions (location search functions that are optional)
        # For now, we'll add them to the base class's free function tracking
        self.compare_class.add_free_function(convs)

        standard_functions = self.get_standard_functions(functions)
        messages = []
        query = convs[0]['content']
        messages.append({"role": "user", "content": query})

        self.init_golden(convs)

        while True:
            # Call model using litellm
            llm_kwargs = {
                "model": self.model_name,
                "max_completion_tokens": self.max_completion_tokens,
                "temperature": 0.0,
                "return_type": "full",
                "tools": standard_functions
            }

            # Add vLLM/local server URL if specified
            if self.vllm_url:
                llm_kwargs["api_base"] = self.vllm_url

                # Workaround: Use OpenAI client directly for vLLM
                try:
                    from openai import OpenAI
                    client = OpenAI(api_key="EMPTY", base_url=self.vllm_url)

                    response = client.chat.completions.create(
                        model=self.model_name.replace("openai/", ""),
                        messages=messages,
                        tools=llm_kwargs.get("tools"),
                        temperature=llm_kwargs.get("temperature", 0.0),
                        max_tokens=llm_kwargs.get("max_completion_tokens", 4096)
                    )
                    llm_response = response
                except ImportError:
                    self.logger.warning("OpenAI package not found, falling back to litellm")
                    llm_response = get_llm_response(messages, **llm_kwargs)
            else:
                llm_response = get_llm_response(messages, **llm_kwargs)

            if llm_response is None:
                return self.return_result(messages, {"error_type": "unknown_error", "content": "llm_response is None"})

            response_message = llm_response.choices[0].message
            tool_calls = response_message.tool_calls

            if tool_calls:
                # Model invoked tool calls
                if self.golden_fcs == []:
                    self.logger.error(f"Output FC:\n{tool_calls}")
                    return self.return_result(messages, {"error_type": "func_hallucination", "content": "Model continued to output function call when expected to stop."})

                # Add assistant message with tool calls
                messages.append(response_message.model_dump())

                # Convert to standard format
                function_calls = [self.get_standard_fc(tool_call) for tool_call in tool_calls]

                # Use BFCL-based comparison (replaces CompareFC)
                self.error_message, success_map, success_matched, format_error = self.compare_turn_prediction(
                    functions, messages[:-1],
                    copy.deepcopy(function_calls), self.golden_fcs,
                    self.golden_obs
                )

                if len(success_map) == 0 and format_error == {}:
                    return self.return_result(messages, self.error_message)

                self.correct_count += len(success_map)

                # Generate observations based on validation results
                real_time_obs = []
                for t in range(len(tool_calls)):
                    if t in success_map:
                        temp_obs = success_map[t]
                    elif t in format_error:
                        temp_obs = format_error[t]
                    else:
                        temp_obs = self.unexpect_call_resp
                    real_time_obs.append(temp_obs)

                # Add tool responses to conversation
                for i, obs in enumerate(real_time_obs):
                    if isinstance(obs, dict):
                        obs_content = json.dumps(obs, ensure_ascii=False)
                    else:
                        obs_content = str(obs)

                    messages.append({
                        "tool_call_id": tool_calls[i].id,
                        "role": "tool",
                        "name": tool_calls[i].function.name,
                        "content": obs_content
                    })

                # Update remaining ground-truth function calls
                self.process_matches(success_matched)

                self.logger.info(f"Function Calls: \n{json.dumps(function_calls, ensure_ascii=False, indent=4)}\n")
                self.logger.info(f"Ground-truth Function Calls: \n{json.dumps(self.golden_fcs, ensure_ascii=False, indent=4)}\n")
                self.logger.info(f"Observations:\n{json.dumps(real_time_obs, ensure_ascii=False, indent=4)}\n")

            elif response_message.content:
                # Text-only response (final answer)
                final_response = response_message.content
                self.logger.info(f"Final Response: {final_response}\n")
                messages.append({"role": "assistant", "content": final_response})
                return self.return_result(messages)

            else:
                return self.return_result(messages, {"error_type": "unknown_error", "content": "Unknown response type"})
