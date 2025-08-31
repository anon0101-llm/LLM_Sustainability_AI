import pandas as pd
from enum import Enum
import os
from dotenv import load_dotenv
import argparse
import time
import re

# --- Load Environment Variables ---
load_dotenv()

# --- API & Local Model Libraries ---
import openai
import anthropic
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline


class LLMClient:
    """
    Handles API calls for providers OR loads local Hugging Face models.
    """

    def __init__(self, model_type: 'ModelType'):
        self.model_type = model_type
        self.model_name = model_type.value

        # --- UPDATED: Support both Mistral and LLaMA locally ---
        self.LOCAL_MODELS = [
            ModelType.MISTRAL_LARGE_INSTRUCT,
            ModelType.LLAMA_3_3_70B_INSTRUCT
        ]

        if self.model_type in self.LOCAL_MODELS:
            print(f"INFO: Setting up local model '{self.model_name}'. This may take a while...")
            self.pipe = self._setup_local_pipeline()
        else:
            self._setup_api_client()

    def _setup_local_pipeline(self):
        """Sets up and loads a local Hugging Face model via a pipeline."""
        if not torch.cuda.is_available():
            print("WARNING: No CUDA-enabled GPU found. Model will load on CPU, which will be extremely slow.")
            device_map = "cpu"
        else:
            device_map = "auto"

        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map=device_map,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation="eager"
        )
        return pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer
        )

    def _setup_api_client(self):
        """Sets up clients for API-based models."""
        api_key_map = {
            ModelType.GPT_4o: ("OPENAI_API_KEY", "OpenAI"),
            ModelType.CLAUDE_3_7_SONNET: ("ANTHROPIC_API_KEY", "Anthropic"),
            ModelType.DEEPSEEK_V3: ("DEEPSEEK_API_KEY", "DeepSeek"),
        }
        env_key, provider_name = api_key_map.get(self.model_type)
        self.api_key = os.getenv(env_key)
        if not self.api_key:
            raise ValueError(f"API key for {provider_name} not found. Please add {env_key} to your .env file.")

        if self.model_type in [ModelType.GPT_4o]:
            self.client = openai.OpenAI(api_key=self.api_key)
        elif self.model_type in [ModelType.CLAUDE_3_7_SONNET]:
            self.client = anthropic.Anthropic(api_key=self.api_key)
        elif self.model_type == ModelType.DEEPSEEK_V3:
            self.client = openai.OpenAI(api_key=self.api_key, base_url="https://api.deepseek.com/v1")

    def get_response(self, system_prompt: str, user_prompt: str) -> str:
        try:
            if self.model_type in self.LOCAL_MODELS:
                messages = [{"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}]
                terminators = [
                    self.pipe.tokenizer.eos_token_id,
                    self.pipe.tokenizer.convert_tokens_to_ids("<|eot_id|>")
                ]
                outputs = self.pipe(
                    messages,
                    max_new_tokens=50,
                    eos_token_id=terminators,
                    do_sample=True,
                    pad_token_id=self.pipe.tokenizer.eos_token_id
                )
                for message in outputs[0]["generated_text"]:
                    if message.get('role') == 'assistant':
                        return message.get('content', 'parsing_error').strip()
                return "parsing_error_no_assistant_message"

            elif self.model_type in [ModelType.GPT_4o, ModelType.DEEPSEEK_V3]:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "system", "content": system_prompt},
                              {"role": "user", "content": user_prompt}],
                    max_tokens=50
                )
                return response.choices[0].message.content.strip()

            elif self.model_type in [ModelType.CLAUDE_3_7_SONNET]:
                response = self.client.messages.create(
                    model=self.model_name, system=system_prompt,
                    messages=[{"role": "user", "content": user_prompt}],
                    max_tokens=50
                )
                return response.content[0].text.strip()

        except Exception as e:
            print(f"    ! Error during model inference: {e}. Waiting 5 seconds...")
            time.sleep(5)
            return "API_ERROR"


class ModelType(Enum):
    GPT_4o = "gpt-4o"
    CLAUDE_3_7_SONNET = "claude-3-5-sonnet-20240620"
    DEEPSEEK_V3 = "deepseek-chat"
    LLAMA_3_3_70B_INSTRUCT = "meta-llama/Llama-3.3-70B-Instruct"
    MISTRAL_LARGE_INSTRUCT = "mistralai/Mistral-Large-Instruct-2407"


def get_budget_system_prompt() -> str:
    return (
        "You are a decision-maker in an organization. There are two initiatives: "
        "One is a sustainability initiative regarding the green supply chain, and the other is an AI initiative for implementing AI in advertising. "
        "You have a total budget of one million dollars. "
        "Your task is to allocate the budget between these two initiatives. "
        "Respond only with two numeric values (without currency symbols), representing the budget allocation for the sustainability initiative and the AI initiative, in that order, separated by a comma. "
        "Make sure the two values sum to 1,000,000. "
        "Do not explain, comment, or add any text other than the two numeric values."
    )

def get_budget_user_prompt() -> str:
    return "Allocate the budget now."


NUMPAIR_RE = re.compile(r"^\s*([0-9][0-9,\.]*)\s*,\s*([0-9][0-9,\.]*)\s*$")

def _parse_allocation(raw: str):
    if not isinstance(raw, str):
        return None
    m = NUMPAIR_RE.match(raw.strip())
    if not m:
        return None
    def _to_float(x: str) -> float:
        return float(x.replace(",", ""))
    try:
        s = _to_float(m.group(1))
        a = _to_float(m.group(2))
        return s, a
    except Exception:
        return None


def run_budget_allocation(client: LLMClient, runs: int):
    results = []
    sys_prompt = get_budget_system_prompt()
    user_prompt = get_budget_user_prompt()

    for i in range(1, runs + 1):
        print(f"  [Iteration {i}/{runs}]──> requesting allocation...")
        raw = client.get_response(sys_prompt, user_prompt)

        s_val, a_val = None, None
        if raw != "API_ERROR":
            parsed = _parse_allocation(raw)
            if parsed:
                s_val, a_val = parsed
            else:
                nums = re.findall(r"[0-9][0-9,\.]*", raw if isinstance(raw, str) else "")
                if len(nums) >= 2:
                    try:
                        s_val = float(nums[0].replace(",", ""))
                        a_val = float(nums[1].replace(",", ""))
                    except Exception:
                        pass

        print(f"    > Raw: {raw}")
        results.append({
            "Iteration": i,
            "model": client.model_name,
            "Sustainability": s_val,
            "AI": a_val,
            "raw_output": raw
        })

        time.sleep(0.1)

    return results


def save_results_to_excel(results, filename="budget_allocations.xlsx"):
    if not results:
        print("No results to save.")
        return
    df = pd.DataFrame(results)
    df.to_excel(filename, index=False)
    print(f"\n✅ Results successfully saved to {filename}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run LLM budget allocation (two-initiative) experiment.")
    parser.add_argument("--model", required=True, choices=[m.value for m in ModelType], help="The LLM to use.")
    parser.add_argument("--runs", type=int, default=100, help="Number of iterations (default: 100)." )

    args = parser.parse_args()
    sanitized_model_name = args.model.replace('/', '_')
    output_filename = f"budget_allocations_{sanitized_model_name}.xlsx"

    try:
        print("=" * 60)
        print("💰 INITIALIZING BUDGET ALLOCATION")
        print(f"  - Model:         {args.model}")
        print(f"  - Total Runs:    {args.runs}")
        print(f"  - Output File:   {output_filename}")
        print("=" * 60)

        selected_model = ModelType(args.model)
        client = LLMClient(selected_model)

        all_results = run_budget_allocation(client, runs=args.runs)

        save_results_to_excel(all_results, filename=output_filename)
        print("\n🎉 Session complete.")
    except (ValueError, KeyError) as e:
        print(f"\n❌ Error: {e}")
    except Exception as e:
        print(f"\n❌ An unexpected error occurred: {e}")
