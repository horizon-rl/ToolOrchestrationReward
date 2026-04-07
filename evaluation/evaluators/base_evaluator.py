import json
import copy

class BaseModelEvaluator:
    """Base class for running model evaluations with function calling capabilities."""

    # Class-level shared CompareFC instance (loads embedding model only once)
    _shared_compare_fc = None
    _compare_fc_initialized = False

    def __init__(self, args, logger, use_compare_fc=True):
        self.logger = logger
        self.args = args

        # Error handling
        self.error_message = None
        self.unexpected_call_response = {
            "api_status": True,
            "content": "There is a problem with your api call, please double-check for possible problems."
        }

        # Initialize comparison utilities (shared across all instances)
        # Only initialize if requested (BFCLToolCallingEvaluator doesn't need it)
        if use_compare_fc:
            if not BaseModelEvaluator._compare_fc_initialized:
                # Import CompareFC only when needed
                try:
                    from utils.compare_method import CompareFC
                except ImportError as e:
                    raise ImportError(
                        "The legacy semantic-matching evaluator is not included in this open-source release. "
                        "Please use BFCL evaluation (`--use-bfcl`) instead."
                    ) from e
                BaseModelEvaluator._shared_compare_fc = CompareFC(args, logger)
                BaseModelEvaluator._compare_fc_initialized = True

            self.compare_class = BaseModelEvaluator._shared_compare_fc
            self.free_function_list = self.compare_class.free_function_list
        else:
            # Create a minimal mock for BFCL evaluator
            self.compare_class = self._create_minimal_compare_class()
            self.free_function_list = ["Location_to_Lat_Long", "Search_Hotel_Destination",
                                      "Search_Attraction_Location", "Search_Car_Location",
                                      "Search_Flight_Location", "Taxi_Search_Location"]

        # Evaluation state
        self._reset_evaluation_state()

    def _create_minimal_compare_class(self):
        """Create a minimal compare class for BFCL evaluator (no embedding model)"""
        class MinimalCompareClass:
            def __init__(self):
                self.free_functions = {}
                self.free_function_list = ["Location_to_Lat_Long", "Search_Hotel_Destination",
                                          "Search_Attraction_Location", "Search_Car_Location",
                                          "Search_Flight_Location", "Taxi_Search_Location"]

            def add_free_function(self, convs):
                """Track free functions from conversations"""
                for i, turn in enumerate(convs):
                    if "function_call" not in turn:
                        continue
                    for j, func_call in enumerate(turn['function_call']):
                        if func_call['name'] in self.free_function_list:
                            if json.dumps(func_call) not in self.free_functions:
                                self.free_functions[json.dumps(func_call)] = {
                                    "called": False,
                                    "obs": convs[i+1]['content'][j]
                                }

        return MinimalCompareClass()
    
    def _reset_evaluation_state(self):
        """Reset all evaluation state variables."""
        self.current_turn_id = 0
        self.correct_count = 0
        self.function_call_chain = []
        self.observation_chain = []
        self.expected_function_calls = []
        self.expected_observations = []

    # Function classification methods
    def _is_free_function_only(self, function_calls):
        """Check if all function calls are 'free' functions that don't count toward evaluation."""
        # Special case for hotel search
        for call in function_calls:
            if (call['name'] == "Search_Hotels" and 
                call.get("arguments", {}).get("search_type") == "hotel"):
                return True

        function_names = {fc["name"] for fc in function_calls}
        return function_names.issubset(set(self.free_function_list))
    
    def _is_location_search_function(self, function_call):
        """Check if function call is a location search function."""
        location_functions = {
            "Search_Hotel_Destination", "Search_Attraction_Location", 
            "Search_Car_Location", "Search_Flight_Location", "Taxi_Search_Location"
        }
        return function_call['name'] in location_functions
    
    def _is_hotel_search_function(self, function_call):
        """Check if function call is a hotel search function."""
        return (function_call['name'] == "Search_Hotels" and 
                function_call.get("arguments", {}).get("search_type") == "hotel")

    # Golden data initialization and management
    def init_golden(self, conversations):
        """Initialize golden reference data from conversation history."""
        self._reset_evaluation_state()
        self._extract_golden_chains(conversations)
        self._validate_golden_chains()
        self._setup_current_turn()

    def _extract_golden_chains(self, conversations):
        """Extract function calls and observations from conversation history."""
        for turn in conversations:
            if "function_call" in turn:
                self.function_call_chain.append(turn['function_call'])
            elif turn['role'] == "observation":
                self.observation_chain.append(turn['content'])

    def _validate_golden_chains(self):
        """Validate that function calls and observations match in length."""
        if len(self.function_call_chain) != len(self.observation_chain):
            raise ValueError(
                f"Function call and observation length mismatch: "
                f"{len(self.function_call_chain)} vs {len(self.observation_chain)}"
            )

    def _setup_current_turn(self):
        """Set up the current turn's expected function calls and observations."""
        if not self.function_call_chain:
            self.expected_function_calls = []
            self.expected_observations = []
            return
            
        self.expected_function_calls = copy.deepcopy(self.function_call_chain[self.current_turn_id])
        self.expected_observations = copy.deepcopy(self.observation_chain[self.current_turn_id])

        # Handle free functions
        if self._is_free_function_only(self.expected_function_calls):
            self._advance_to_next_turn()

    def _advance_to_next_turn(self):
        """Move to the next turn if available."""
        self.current_turn_id += 1
        if self.current_turn_id < len(self.function_call_chain):
            self.expected_function_calls.extend(
                copy.deepcopy(self.function_call_chain[self.current_turn_id])
            )
            self.expected_observations.extend(
                copy.deepcopy(self.observation_chain[self.current_turn_id])
            )

    # Success calculation
    def _calculate_success_turn(self, remaining_function_calls, total_function_calls):
        """Calculate how many turns were successfully completed."""
        remaining_turn_ids = []
        
        for turn_idx, fc_list in enumerate(total_function_calls):
            for remaining_fc in remaining_function_calls:
                if remaining_fc in fc_list:
                    remaining_turn_ids.append(turn_idx)
        
        if not remaining_turn_ids:
            return len(total_function_calls)
        
        return max(min(remaining_turn_ids), 0)

    # Match processing
    def process_matches(self, successful_matches):
        """Process successfully matched function calls and update state."""
        self._remove_matched_calls(successful_matches)
        
        if successful_matches:
            self._advance_to_next_turn()
        
        self._handle_free_function_matches()
        
        # Handle free functions in remaining expected calls
        if self._is_free_function_only(self.expected_function_calls):
            self._advance_to_next_turn()

    def _remove_matched_calls(self, successful_matches):
        """Remove successfully matched calls from expected lists."""
        for matched_call in successful_matches:
            if matched_call in self.expected_function_calls:
                call_index = self.expected_function_calls.index(matched_call)
                self.expected_observations.pop(call_index)
                self.expected_function_calls.remove(matched_call)

    def _handle_free_function_matches(self):
        """Handle matches for free functions."""
        for key, value in self.compare_class.free_functions.items():
            if value['called'] and json.loads(key) in self.expected_function_calls:
                matched_call = json.loads(key)
                call_index = self.expected_function_calls.index(matched_call)
                self.expected_observations.pop(call_index)
                self.expected_function_calls.remove(matched_call)

    # Result handling
    def return_result(self, messages, error_info=None):
        """Return the final evaluation result."""
        if error_info:
            success_turn = self._calculate_success_turn(
                self.expected_function_calls, 
                self.function_call_chain
            )
            return messages, error_info, success_turn, self.correct_count

        return self._handle_successful_completion(messages)

    def _handle_successful_completion(self, messages):
        """Handle the case where execution completed without errors."""
        self._clean_remaining_free_functions()
        
        # Check if evaluation is complete
        has_remaining_turns = self.current_turn_id < len(self.function_call_chain)
        has_remaining_calls = len(self.expected_function_calls) > 0
        
        if has_remaining_turns or has_remaining_calls:
            self.logger.info(f"Turn ID: {self.current_turn_id}, Total turns: {len(self.function_call_chain)}")
            self.logger.info(f"Remaining expected calls: {self.expected_function_calls}")
            
            error_info = {"error_type": "incomplete_execution", "content": "Execution stopped before completing all expected function calls."}
            return self.return_result(messages, error_info)
        
        elif len(self.expected_function_calls) == 0:
            return messages, "Success.", len(self.function_call_chain), self.correct_count
        
        else:
            raise RuntimeError("Unexpected evaluation state encountered.")

    def _clean_remaining_free_functions(self):
        """Remove free functions from remaining expected calls."""
        calls_to_remove = []
        
        for call in self.expected_function_calls:
            if (self._is_hotel_search_function(call) or 
                self._is_location_search_function(call)):
                calls_to_remove.append(call)
        
        for call in calls_to_remove:
            self.expected_function_calls.remove(call)

    # Properties for backward compatibility
    @property
    def golden_fcs(self):
        """Backward compatibility property."""
        return self.expected_function_calls
    
    @property 
    def golden_obs(self):
        """Backward compatibility property."""
        return self.expected_observations
    
    @property
    def turn_id(self):
        """Backward compatibility property."""
        return self.current_turn_id
    
    @property
    def fc_chain(self):
        """Backward compatibility property."""
        return self.function_call_chain
    
    @property
    def obs_chain(self):
        """Backward compatibility property."""
        return self.observation_chain
    
    @property
    def CompareClass(self):
        """Backward compatibility property."""
        return self.compare_class
    
    @property
    def unexpect_call_resp(self):
        """Backward compatibility property."""
        return self.unexpected_call_response