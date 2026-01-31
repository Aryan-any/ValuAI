# from litellm import completion
# from dotenv import load_dotenv
# import os

# load_dotenv(override=True)

# DEFAULT_MODEL_NAME = os.getenv("PRICER_PREPROCESSOR_MODEL", "ollama/tinyllama")
# DEFAULT_REASONING_EFFORT = "low" if "gpt-oss" in DEFAULT_MODEL_NAME else None

# SYSTEM_PROMPT = """Create a concise description of a product. Respond only in this format. Do not include part numbers.
# Title: Rewritten short precise title
# Category: eg Electronics
# Brand: Brand name
# Description: 1 sentence description
# Details: 1 sentence on features"""


# class Preprocessor:
#     def __init__(
#         self,
#         model_name=DEFAULT_MODEL_NAME,
#         reasoning_effort=DEFAULT_REASONING_EFFORT,
#         base_url=None,
#     ):
#         self.total_input_tokens = 0
#         self.total_output_tokens = 0
#         self.total_cost = 0
#         self.model_name = model_name
#         self.reasoning_effort = reasoning_effort
#         self.base_url = base_url
#         if "ollama" in model_name and not base_url:
#             self.base_url = "http://localhost:11434"

#     def messages_for(self, text: str) -> list[dict]:
#         return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": text}]




        

#     # def preprocess(self, text: str) -> str:
#     #     messages = self.messages_for(text)
#     #     response = completion(
#     #         messages=messages,
#     #         model=self.model_name,
#     #         reasoning_effort=self.reasoning_effort,
#     #         api_base=self.base_url,
#     #     )
#     #     self.total_input_tokens += response.usage.prompt_tokens
#     #     self.total_output_tokens += response.usage.completion_tokens
#     #     self.total_cost += response._hidden_params["response_cost"]
#     #     return response.choices[0].message.content



    

# #     def preprocess(self, text: str) -> str:
# #     messages = self.messages_for(text)

# #     try:
# #         response = completion(
# #             messages=messages,
# #             model=self.model_name,
# #             reasoning_effort=self.reasoning_effort,
# #             api_base=self.base_url,
# #             timeout=30,
# #         )

# #     except Exception as e:
# #         # Hard fallback if Ollama crashes (OOM, CUDA failure, etc.)
# #         print(f"[Preprocessor] Primary model failed: {e}")
# #         print("[Preprocessor] Falling back to CPU-safe model")

# #         response = completion(
# #             messages=messages,
# #             model="ollama/tinyllama",
# #             api_base="http://localhost:11434",
# #             temperature=0,
# #             timeout=20,
# # )


# #     self.total_input_tokens += response.usage.prompt_tokens
# #     self.total_output_tokens += response.usage.completion_tokens
# #     self.total_cost += response._hidden_params.get("response_cost", 0)

# #     return response.choices[0].message.content




# def preprocess(self, text: str) -> str:
#     """
#     Disable LLM preprocessing.
#     Ollama on Windows is unstable with CUDA.
#     Return raw text to keep pipeline functional.
#     """
#     return text


from dotenv import load_dotenv
import os

load_dotenv(override=True)

SYSTEM_PROMPT = """Create a concise description of a product. Respond only in this format. Do not include part numbers.
Title: Rewritten short precise title
Category: eg Electronics
Brand: Brand name
Description: 1 sentence description
Details: 1 sentence on features
"""


class Preprocessor:
    def __init__(self):
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost = 0
        # self.model_name = "disabled"

    def messages_for(self, text: str) -> list[dict]:
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ]

    def preprocess(self, text: str) -> str:
        """
        LLM preprocessing intentionally disabled.

        Reason:
        - Ollama on Windows is unstable (CUDA / memory issues)
        - Preprocessing is non-critical for pricing accuracy
        - Keeps ValuAI pipeline deterministic and stable
        """
        return text
