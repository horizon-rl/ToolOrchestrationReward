"""
BFCL Tool Client Manager for BookingAPI integration.

This client manager allows verl training to execute BookingAPI functions
from the BFCL evaluation environment instead of StableToolBench APIs.
"""

import sys
import os
import json
import asyncio
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import logging

# Add BFCL to path to import BookingAPI
_script_dir = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.getenv("REPO_ROOT", os.path.abspath(os.path.join(_script_dir, "..", "..")))
bfcl_path = os.path.join(_repo_root, "BFCL")
if bfcl_path not in sys.path:
    sys.path.insert(0, bfcl_path)

from bfcl_eval.eval_checker.multi_turn_eval.func_source_code.booking_api import BookingAPI

from verl.tools.utils.mcp_clients.utils import TokenBucket

logger = logging.getLogger(__name__)


@dataclass
class ToolResult:
    """Represents the result of a tool call"""
    content: List[Dict[str, Any]]
    isError: bool = False


class BFCLToolClient:
    """Client for BFCL BookingAPI tool calling"""

    def __init__(self, tools: List[Dict[str, Any]] = None):
        """
        Initialize BFCL tool client.

        Args:
            tools: List of tool schemas in OpenAI function format
        """
        self.tools = tools or []
        self.booking_api = BookingAPI()

        # Build function name to method mapping
        self.function_map = {}
        for tool in self.tools:
            if tool['type'] == 'function':
                func_name = tool['function']['name']
                # Map function name to BookingAPI method
                if hasattr(self.booking_api, func_name):
                    self.function_map[func_name] = getattr(self.booking_api, func_name)
                else:
                    logger.warning(f"Function {func_name} not found in BookingAPI")

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass

    async def get_tools(self) -> List[Dict[str, Any]]:
        """Return tools in OpenAI format"""
        return self.tools

    async def call_tool(self, tool_name: str, parameters: Dict[str, Any]) -> ToolResult:
        """
        Call a BFCL BookingAPI function.

        Args:
            tool_name: Function name (e.g., "Search_Hotels")
            parameters: Function arguments as dict

        Returns:
            ToolResult with the API response
        """
        try:
            if tool_name not in self.function_map:
                error_msg = f"Function '{tool_name}' not found in BookingAPI"
                logger.error(error_msg)
                return ToolResult(
                    content=[{
                        "type": "text",
                        "text": json.dumps({
                            "error": error_msg
                        })
                    }],
                    isError=True
                )

            # Get the method
            method = self.function_map[tool_name]

            logger.debug(f"Calling BookingAPI.{tool_name} with parameters: {parameters}")

            # Execute the function (BookingAPI methods are synchronous)
            # Run in executor to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: method(**parameters)
            )

            # Check if result is an error
            is_error = isinstance(result, dict) and "error" in result

            if is_error:
                logger.warning(f"BookingAPI.{tool_name} returned error: {result}")
            else:
                logger.info(f"BookingAPI.{tool_name} executed successfully")

            return ToolResult(
                content=[{
                    "type": "text",
                    "text": json.dumps(result, ensure_ascii=False)
                }],
                isError=is_error
            )

        except TypeError as e:
            # Parameter mismatch error
            error_msg = f"Parameter error calling {tool_name}: {str(e)}"
            logger.error(error_msg)
            return ToolResult(
                content=[{
                    "type": "text",
                    "text": json.dumps({
                        "error": error_msg
                    })
                }],
                isError=True
            )
        except Exception as e:
            # Other execution errors
            error_msg = f"Error executing {tool_name}: {str(e)}"
            logger.error(error_msg)
            return ToolResult(
                content=[{
                    "type": "text",
                    "text": json.dumps({
                        "error": error_msg
                    })
                }],
                isError=True
            )

    def call_tool_sync(self, tool_name: str, parameters: Dict[str, Any]) -> ToolResult:
        """
        Synchronous version of call_tool for evaluation purposes.

        Args:
            tool_name: Function name (e.g., "Search_Hotels")
            parameters: Function arguments as dict

        Returns:
            ToolResult with the API response
        """
        try:
            if tool_name not in self.function_map:
                error_msg = f"Function '{tool_name}' not found in BookingAPI"
                logger.error(error_msg)
                return ToolResult(
                    content=[{
                        "type": "text",
                        "text": json.dumps({
                            "error": error_msg
                        })
                    }],
                    isError=True
                )

            # Get the method
            method = self.function_map[tool_name]

            logger.debug(f"Calling BookingAPI.{tool_name} with parameters: {parameters}")

            # Execute the function synchronously (BookingAPI methods are synchronous)
            result = method(**parameters)

            # Check if result is an error
            is_error = isinstance(result, dict) and "error" in result

            if is_error:
                logger.warning(f"BookingAPI.{tool_name} returned error: {result}")
            else:
                logger.info(f"BookingAPI.{tool_name} executed successfully")

            return ToolResult(
                content=[{
                    "type": "text",
                    "text": json.dumps(result, ensure_ascii=False)
                }],
                isError=is_error
            )

        except TypeError as e:
            # Parameter mismatch error
            error_msg = f"Parameter error calling {tool_name}: {str(e)}"
            logger.error(error_msg)
            return ToolResult(
                content=[{
                    "type": "text",
                    "text": json.dumps({
                        "error": error_msg
                    })
                }],
                isError=True
            )
        except Exception as e:
            # Other execution errors
            error_msg = f"Error executing {tool_name}: {str(e)}"
            logger.error(error_msg)
            return ToolResult(
                content=[{
                    "type": "text",
                    "text": json.dumps({
                        "error": error_msg
                    })
                }],
                isError=True
            )


class BFCLToolClientManager:
    """Client manager for BFCL BookingAPI"""

    def __init__(self):
        self.initialized = False
        self.client = None
        self.rate_limiter = None
        self.tools = []

    async def initialize(self,
                        tool_schema_path: str,
                        rate_limit: float = 10.0):
        """
        Initialize the BFCL client manager.

        Args:
            tool_schema_path: Path to booking_api.json function schemas
            rate_limit: Max API calls per second
        """
        if self.initialized:
            return

        # Expand environment variables in path (fallback for ${VAR} syntax)
        tool_schema_path = os.path.expandvars(tool_schema_path)

        # Load tool schemas from booking_api.json (JSONL format)
        tool_schemas = []
        with open(tool_schema_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                # Parse each line as a separate JSON object
                schema = json.loads(line)

                # Convert to OpenAI function calling format
                tool_schema = {
                    'type': 'function',
                    'function': {
                        'name': schema['name'],
                        'description': schema['description'],
                        'parameters': schema['parameters']
                    }
                }
                tool_schemas.append(tool_schema)

        self.tools = tool_schemas

        # Create client
        self.client = BFCLToolClient(tools=self.tools)

        # Setup rate limiting
        self.rate_limiter = TokenBucket(rate_limit)
        self.initialized = True

        logger.info(f"Initialized BFCLToolClientManager with {len(self.tools)} tools from {tool_schema_path}")

    async def fetch_tool_schemas(self, tool_selected_list=None) -> List[Dict[str, Any]]:
        """
        Get tools in OpenAI format.

        Args:
            tool_selected_list: Optional list of specific tool names to return

        Returns:
            List of tool schemas
        """
        if tool_selected_list is None:
            return self.tools

        # Filter to only requested tools
        tool_names = set(tool_selected_list)
        return [
            tool for tool in self.tools
            if tool['type'] == 'function' and tool['function']['name'] in tool_names
        ]

    async def call_tool(self, tool_name: str, parameters: Dict[str, Any], timeout: Optional[float] = None):
        """
        Call a tool with rate limiting.

        Args:
            tool_name: Function name
            parameters: Function arguments
            timeout: Optional timeout (not currently used)

        Returns:
            ToolResult
        """
        if not self.client:
            return ToolResult(
                content=[{"type": "text", "text": "Client not initialized"}],
                isError=True
            )

        # Apply rate limiting
        while not self.rate_limiter.acquire():
            await asyncio.sleep(0.1)

        async with self.client:
            return await self.client.call_tool(tool_name, parameters)


# Create global client manager instance
ClientManager = BFCLToolClientManager()
