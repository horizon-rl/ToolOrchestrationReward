# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
BFCL Tool wrapper for BookingAPI integration with verl training.
"""

import json
import logging
import os
from typing import Any, Optional
from uuid import uuid4

from verl.tools.utils.mcp_clients.BFCLClientManager import ClientManager
from verl.utils.rollout_trace import rollout_trace_op

from .base_tool import BaseTool
from .schemas import OpenAIFunctionToolSchema

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class BFCLTool(BaseTool):
    """Tool wrapper for BFCL BookingAPI functions."""

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instance_dict = {}
        self.timeout = config.get("timeout", 30)

        logger.info(f"Initialized BFCLTool with config: {config}")

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        """Return the OpenAI tool schema."""
        return self.tool_schema

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> str:
        """Create a tool instance.

        Args:
            instance_id: The instance id of the tool.

        Returns:
            The instance id of the tool.
        """
        if instance_id is None:
            instance_id = str(uuid4())
        self._instance_dict[instance_id] = {
            "response": "",
            "reward": [],
        }
        return instance_id

    async def _call_tool(self, instance_id, parameters) -> tuple[str, dict]:
        """
        Call the BFCL BookingAPI function.

        Args:
            instance_id: Instance ID
            parameters: Function parameters

        Returns:
            Tuple of (result_text, metadata)
        """
        err_msg = ""
        try:
            call_tool_result = await ClientManager.call_tool(self.name, parameters, self.timeout)
        except Exception as e:
            err_msg = f"\n Tool call failed: {e}"
            # Create error result
            call_tool_result = type('obj', (object,), {
                'content': [{'type': 'text', 'text': json.dumps({"error": str(e)})}],
                'isError': True
            })()

        logger.debug(f"Tool result for instance {instance_id} with tool {self.name}: {call_tool_result.content}")
        result, metadata = self._parse_tool_result(call_tool_result.content)
        metadata["api_request_error"] = err_msg
        metadata["is_error"] = call_tool_result.isError
        return result, metadata

    @rollout_trace_op
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[str, float, dict]:
        """
        Execute the BFCL tool.

        Args:
            instance_id: Instance ID
            parameters: Function parameters
            **kwargs: Additional arguments

        Returns:
            Tuple of (result_text, reward, metrics)
        """
        if self.name == "" or self.name is None or parameters is None:
            error_msg = "Error: 'parameters' is missing or empty."
            logger.error(f"[BFCLTool] {error_msg} Received tool name: {self.name}, parameters: {parameters}")
            return json.dumps({"error": error_msg}), 0.0, {}

        try:
            result_text, metadata = await self._call_tool(instance_id, parameters)

            # Store results in instance dictionary
            self._instance_dict[instance_id]["reward"].append(result_text.strip())

            # Convert metadata to metrics
            metrics = {
                "is_error": metadata.get("is_error", False),
                "api_request_error": metadata.get("api_request_error", ""),
            }

            return result_text, 0.0, metrics

        except Exception as e:
            error_result = json.dumps({"error": f"Tool execution failed: {e}"})
            logger.error(f"[BFCLTool] Execution failed: {e}")
            return error_result, 0.0, {"error": str(e)}

    async def calc_reward(self, instance_id: str, **kwargs) -> str:
        """Get the accumulated rewards for this instance."""
        return self._instance_dict[instance_id]["reward"]

    async def release(self, instance_id: str, **kwargs) -> None:
        """Release the instance and clean up resources."""
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]

    def _parse_tool_result(self, content: list) -> tuple[str, dict]:
        """
        Parse tool result from BFCL response format.

        Args:
            content: List of content dicts with 'type' and 'text' fields

        Returns:
            Tuple of (text_result, metadata_dict)
        """
        tools_content = [part['text'] for part in filter(lambda x: x['type'] == "text", content)]
        return " ".join(tools_content), {}
