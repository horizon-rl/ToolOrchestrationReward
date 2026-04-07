import copy
import json
from utils.utils import get_llm_response
from evaluators.base_evaluator import BaseModelEvaluator


class ToolCallingEvaluator(BaseModelEvaluator):
    def __init__(self, args, logger):
        super().__init__(args, logger)
        self.model_name = args.model_name
        self.max_completion_tokens = getattr(args, 'max_completion_tokens', 4096)
        self.vllm_url = getattr(args, 'vllm_url', None)
    
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
    
    def run(self, data):
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
        """Internal implementation of run() - can raise exceptions that will be caught by run()"""
        convs, functions = data['conversations'], data['functions']
        self.CompareClass.add_free_function(convs)

        standard_functions = self.get_standard_functions(functions)  # Convert functions to standard input format
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

                # Workaround: Use OpenAI client directly for vLLM to avoid litellm format issues
                try:
                    from openai import OpenAI
                    client = OpenAI(api_key="EMPTY", base_url=self.vllm_url)

                    response = client.chat.completions.create(
                        model=self.model_name.replace("openai/", ""),  # Remove openai/ prefix
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

            if tool_calls:  # Case 1: models invoke tool calls in response.
                if self.golden_fcs == []:
                    self.logger.error(f"Output FC:\n{tool_calls}")
                    return self.return_result(messages, {"error_type": "func_hallucination", "content": "`self.golden_fcs == []`. Expected to stop. But Model continue to output function call."})
                
                # Add assistant message with tool calls to conversation
                # messages.append({"role": "assistant", "tool_calls": function_calls})
                messages.append(response_message.model_dump())
                
                # [Environment] Compare function calls with the ground-truth (i.e., the API executor)
                # Convert tool call result to the standard format which can be used by the environment
                function_calls = [self.get_standard_fc(tool_call) for tool_call in tool_calls]
                # Comparison
                self.error_message, success_map, success_matched, format_error = self.CompareClass.compare_turn_prediction(
                    functions, messages[:-1], 
                    copy.deepcopy(function_calls), self.golden_fcs, 
                    self.golden_obs
                )
                
                if len(success_map) == 0 and format_error == {}:
                    return self.return_result(messages, self.error_message)
                self.correct_count += len(success_map)

                # [Environment] Generate observations based on function calls
                # Decide function call observations by whether it is successfully matched to the ground-truth (by self.CompareClass.compare_turn_prediction)
                real_time_obs = []
                for t in range(len(tool_calls)):
                    if t in success_map:    # matched
                        temp_obs = success_map[t]   
                    elif t in format_error: # format error
                        temp_obs = format_error[t]
                    else:                   # other error
                        temp_obs = self.unexpect_call_resp
                        
                    real_time_obs.append(temp_obs)

                # Add tool responses to conversation history
                for i, obs in enumerate(real_time_obs):
                    # OpenAI API requires tool response content to be a string, not a dict
                    if isinstance(obs, dict):
                        obs_content = json.dumps(obs, ensure_ascii=False)
                    else:
                        obs_content = str(obs)

                    messages.append(
                        {
                            "tool_call_id": tool_calls[i].id,
                            "role": "tool",
                            "name": tool_calls[i].function.name,
                            "content": obs_content
                        }
                    )
                
                # [Environment] Update remaining ground-truth function calls and related responses.
                self.process_matches(success_matched)
                    
                self.logger.info(f"Function Calls: \n{json.dumps(function_calls, ensure_ascii=False, indent=4)}\n")
                self.logger.info(f"Ground-truth Function Calls: \n{json.dumps(self.golden_fcs, ensure_ascii=False, indent=4)}\n")
                self.logger.info(f"Observations:\n{json.dumps(real_time_obs, ensure_ascii=False, indent=4)}\n")

            elif response_message.content:  # Case 2: model only outputs text message in response.
                # Only text output -- final response
                final_response = response_message.content
                self.logger.info(f"Final Response: {final_response}\n")
                messages.append({"role": "assistant", "content": final_response})
                return self.return_result(messages)

            else:
                return self.return_result(messages, {"error_type": "unknown_error", "content": "Unknown response type"})