import json
import os

from bfcl.model_handler.base_handler import BaseHandler
from bfcl.model_handler.constant import GORILLA_TO_OPENAPI
from bfcl.model_handler.constant import DEFAULT_SYSTEM_PROMPT
from bfcl.model_handler.model_style import ModelStyle
from bfcl.model_handler.utils import (
    convert_to_tool,
    func_doc_language_specific_pre_processing,
    system_prompt_pre_processing_chat_model,
)
from openai import OpenAI

class GoGoAgentHandler(BaseHandler):
    def __init__(self, model_name, temperature) -> None:
        super().__init__(model_name, temperature)
        self.model_style = ModelStyle.OpenAI
        self.base_url = "https://api.gogoagent.ai/query_vllm"
        self.client = OpenAI(base_url=self.base_url, api_key=os.getenv("GOGOAGENT_API_KEY"))

    def decode_ast(self, result, language="Python"):
        decoded_output = []
        for invoked_function in result:
            name = list(invoked_function.keys())[0]
            try:
                params = json.loads(invoked_function[name])
            except:
                params = invoked_function[name]
            decoded_output.append({name: params})

        return decoded_output

    # TODO how to use the util version of this?
    def convert_to_function_call(self, function_call_list):
        if type(function_call_list) == dict:
            function_call_list = [function_call_list]
        execution_list = []
        for function_call in function_call_list:
            for key, value in function_call.items():
                try:
                    execution_list.append(
                        f"{key}({','.join([f'{k}={repr(v)}' for k,v in json.loads(value).items()])})"
                    )
                except:
                    execution_list.append(
                        f"{key}({','.join([f'{k}={repr(v)}' for k,v in value.items()])})"
                    )

        return execution_list

    def decode_execute(self, result):
        # TODO use util version of this
        function_call = self.convert_to_function_call(result)
        return function_call

    def doldecode_ast(self, result, language="Python"):
        decoded_output = []
        for invoked_function in result:
            name = invoked_function["name"]
            params = invoked_function["arguments"]
            decoded_output.append({name: params})
        return decoded_output

    ### NON FC methods *** 
    def _pre_query_processing_prompting(self, test_entry: dict) -> dict:
        functions: list = test_entry["function"]
        test_category: str = test_entry["id"].rsplit("_", 1)[0]

        functions = func_doc_language_specific_pre_processing(functions, test_category)

        test_entry["question"][0] = system_prompt_pre_processing_chat_model(
            test_entry["question"][0], DEFAULT_SYSTEM_PROMPT, functions
        )

        return {"message": [], "function": functions}

    def _parse_query_response_prompting(self, api_response: any) -> dict:
        try:
            model_responses = [
                {func_call.name: func_call.arguments}
                for func_call in api_response.choices[0].message.tool_calls
            ]
            tool_call_ids = [
                func_call.id for func_call in api_response.choices[0].message.tool_calls
            ]
            tool_call_func_names = [
                func_call.name
                for func_call in api_response.choices[0].message.tool_calls
            ]
        except:
            model_responses = api_response.choices[0].message.content
            tool_call_ids = []
            tool_call_func_names = []

        model_responses_message_for_chat_history = api_response.choices[0].message

        x = {
            "model_responses": model_responses,
            "model_responses_message_for_chat_history": model_responses_message_for_chat_history,
            "tool_call_ids": tool_call_ids,
            "tool_call_func_names": tool_call_func_names,
            "input_token": 0,
            "output_token": 0
        }
        return x

    def add_first_turn_message_prompting(
        self, inference_data: dict, first_turn_message: list[dict]
    ) -> dict:
        inference_data["message"].extend(first_turn_message)
        return inference_data

    def _add_next_turn_user_message_prompting(
        self, inference_data: dict, user_message: list[dict]
    ) -> dict:
        inference_data["message"].extend(user_message)
        return inference_data

    def _add_assistant_message_prompting(
        self, inference_data: dict, model_response_data: dict
    ) -> dict:
        inference_data["message"].append(
            {"role": "assistant", "content": model_response_data["model_responses"]}
        )
        return inference_data

    def _add_execution_results_prompting(
        self, inference_data: dict, execution_results: list[str], model_response_data: dict
    ) -> dict:
        for execution_result, decoded_model_response in zip(
            execution_results, model_response_data["model_responses_decoded"]
        ):
            inference_data["message"].append(
                {
                    "role": "tool",
                    "name": decoded_model_response,
                    "content": execution_result,
                }
            )

        return inference_data

    def convert_to_dict(self, input_str):
        """
        Convert a JSON-formatted string into a dictionary of tool calls and their arguments.

        Parameters:
        - input_str (str): A JSON-formatted string containing 'tool_calls' with 'name' and 'arguments'.

        Returns:
        - list[dict]: A list of dictionaries with tool call names as keys and their arguments as values.
        """
        try:
            data = json.loads(input_str)
        except json.JSONDecodeError:
            return input_str

        tool_calls = data if isinstance(data, list) else data.get("tool_calls", [])

        result_list = [
            {tool_call.get("name", ""): tool_call.get("arguments", {})}
            for tool_call in tool_calls
            if isinstance(tool_call, dict)
        ]

        return result_list

    def _query_prompting(self, inference_data: dict):
        function: list[dict] = inference_data["function"]
        message: list[dict] = inference_data["message"]

        inference_data["inference_input_log"] = {
            "message": repr(message),
            "function": function,
        }

        api_response = self.client.chat.completions.create(messages=message, tools=function, model=self.model_name)

        return api_response