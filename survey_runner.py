import pandas as pd
from enum import Enum
import os
from dotenv import load_dotenv
import argparse
import time

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
            torch_dtype=torch.bfloat16,   # good default on modern GPUs / falls back if unsupported
            trust_remote_code=True,
            attn_implementation="eager"   # --- Better compatibility (Mistral/LLaMA)
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
        """
        Gets a single response from the configured LLM (local or API).
        """
        try:
            # --- Local inference (Mistral/LLaMA) ---
            if self.model_type in self.LOCAL_MODELS:
                messages = [{"role": "system", "content": system_prompt},
                            {"role": "user",  "content": user_prompt}]

                # Common terminators for instruction-tuned Mistral/LLaMA
                terminators = [
                    self.pipe.tokenizer.eos_token_id,
                    self.pipe.tokenizer.convert_tokens_to_ids("<|eot_id|>")
                ]

                outputs = self.pipe(
                    messages,
                    max_new_tokens=50,
                    eos_token_id=terminators,
                    do_sample=True,
                    pad_token_id=self.pipe.tokenizer.eos_token_id  # avoid pad warning
                )
                for message in outputs[0]["generated_text"]:
                    if message.get('role') == 'assistant':
                        return message.get('content', 'parsing_error').strip()
                return "parsing_error_no_assistant_message"

            # --- API-based logic ---
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


def get_system_prompt(likert_scale: list, main_question: str = None) -> str:
    likert_scale_str = "\n".join(likert_scale)
    constraints = (
        "To prevent elaboration beyond the provided Likert scale, follow these strict constraints:\n"
        "(1) Do not elaborate on your reasoning.\n"
        "(2) Do not say any other things instead of options.\n"
        "(3) Do not apologize.\n"
        "(4) Do not include any 'note' or 'disclaimer'.\n"
        "(5) Never use words like 'cannot', 'unable', 'instead', 'as', 'however', 'it', 'unfortunately', or 'important'.\n"
        "(6) Do not include any negative sentences on the subject of the prompt.\n"
    )

    if main_question:
        task_instruction = (
            f"{main_question}\n"
            f"You must respond with only one of the following options. "
            f"Do not add any other text or explanation.\n\n"
            f"{constraints}\n"
            f"Options:\n{likert_scale_str}"
        )
    else:
        task_instruction = (
            f"For the following statement, you must respond with only one of the following options. "
            f"Do not add any other text or explanation.\n\n"
            f"{constraints}\n"
            f"Options:\n{likert_scale_str}"
        )

    return task_instruction


def _execute_questions(client: LLMClient, survey_name: str, questions: list, system_prompt: str, run_number: int):
    results = []
    total_questions = len(questions)
    for i, question_text in enumerate(questions, 1):
        print(f"  [Question {i}/{total_questions}]──> {question_text[:70]}...")
        response = client.get_response(system_prompt, question_text)

        while response == "API_ERROR":
            print("  Retrying last request...")
            response = client.get_response(system_prompt, question_text)

        print(f"    > Received response: '{response}'")
        results.append({
            "model": client.model_name,
            "survey": survey_name,
            "question": question_text,
            "run_number": run_number,
            "response": response
        })
    return results


# --------- Surveys (culture-free) ---------
def run_aispi_survey(client: LLMClient, run_number: int):
    statements = [
        "AI can help optimize resource use and reduce waste.",
        "AI will create more jobs than it will eliminate.",
        "The energy consumption of AI systems could hinder sustainability efforts.",
        "AI is essential for monitoring and achieving sustainability goals.",
        "The pursuit of AI advancement and sustainability are competing priorities.",
        "AI and sustainability efforts can be mutually reinforcing.",
        "Sustainable development will limit AI advancements.",
        "There are many conflicts between the advancement of AI and sustainability efforts.",
        "AI will hinder sustainable development.",
        "AI will advance sustainable development.",
        "AI and sustainable development go along very well.",
        "It is important for society to integrate both AI advancement and sustainability efforts.",
        "Sustainable development will advance the development of AI."
    ]
    likert_scale = ['Strongly Disagree', 'Disagree', 'Somewhat Disagree', 'Somewhat Agree', 'Agree', 'Strongly Agree']
    system_prompt = get_system_prompt(likert_scale)
    return _execute_questions(client, "AISPI", statements, system_prompt, run_number)


def run_sdg17_survey(client: LLMClient, run_number: int):
    areas = [
        "Global poverty", "World hunger", "Public health", "Education", "Gender equality", "Water security",
        "Renewable energies", "Economic growth", "Innovative industries", "Social inequality",
        "Sustainable cities and communities", "Consumption and production", "Climate action", "Ocean protection",
        "Ecosystem conservation", "Peace and justice", "International cooperation"
    ]
    likert_scale = ['Very negative impact', 'Negative impact', 'Slightly negative impact',
                    'Slightly positive impact', 'Positive impact', 'Very positive impact']
    main_question = "How do you think AI will impact the following areas in the next 10 years?"
    system_prompt = get_system_prompt(likert_scale, main_question)
    results = _execute_questions(client, "SDG17", areas, system_prompt, run_number)
    for res in results:
        res["question"] = f"{main_question} - {res['question']}"
    return results


def run_sdg18_survey(client: LLMClient, run_number: int):
    question = "In your opinion, which of both transformations is more important? (AI vs. Sustainability)"
    likert_scale = ['1 – AI is much more important', '2 – AI is more important', '3 – AI is slightly more important',
                    '4 – Sustainability is slightly more important', '5 – Sustainability is more important',
                    '6 – Sustainability is much more important']
    system_prompt = get_system_prompt(likert_scale, question)
    return _execute_questions(client, "SDG18", [question], system_prompt, run_number)


def run_sdg19_survey(client: LLMClient, run_number: int):
    question = "Do you believe AI and sustainable development will become more integrated in the future?"
    likert_scale = ['Definitely not', 'Probably not', 'Possibly not', 'Possibly yes', 'Probably yes', 'Yes, for sure']
    system_prompt = get_system_prompt(likert_scale, question)
    return _execute_questions(client, "SDG19", [question], system_prompt, run_number)


def run_additional_question_1(client: LLMClient, run_number: int):
    question = "Do you think governments, industries and organizations are doing enough to ensure AI and sustainable development go along with each other?"
    likert_scale = ['1 = Not at all', '2 = Slightly', '3 = Somewhat', '4 = Moderately', '5 = Mostly',
                    '6 = Yes, absolutely']
    system_prompt = get_system_prompt(likert_scale, question)
    return _execute_questions(client, "Additional Question 1", [question], system_prompt, run_number)


def run_additional_questions_2_3(client: LLMClient, run_number: int):
    results = []
    organizations = ["National universities", "International Research Organizations", "Technology companies",
                     "Government", "Non-governmental organizations (NGO)"]

    q2_main = "Who do you think is responsible to ensure AI advancement and sustainable development go along with each other?"
    q2_likert = ["Responsible", "Not Responsible"]
    q2_prompt = get_system_prompt(q2_likert, "For each of the following, state if they are responsible...")
    q2_results = _execute_questions(client, "Additional Question 2", organizations, q2_prompt, run_number)
    for res in q2_results:
        res["question"] = f"{q2_main} - {res['question']}"
    results.extend(q2_results)

    q3_main = "How much confidence do you have in the following to develop and use AI in the best interest of sustainable development?"
    q3_likert = ['1= Most likely', '2= Likely', '3=Somewhat likely', '4=Somewhat unlikely', '5=Unlikely',
                 '6=Definitely not']
    q3_prompt = get_system_prompt(q3_likert, q3_main)
    q3_results = _execute_questions(client, "Additional Question 3", organizations, q3_prompt, run_number)
    for res in q3_results:
        res["question"] = f"{q3_main} - {res['question']}"
    results.extend(q3_results)
    return results


def run_survey(client: LLMClient, survey_name: str, run_number: int):
    survey_functions = {
        "AISPI": run_aispi_survey,
        "SDG17": run_sdg17_survey,
        "SDG18": run_sdg18_survey,
        "SDG19": run_sdg19_survey,
        "AQ1": run_additional_question_1,
        "AQ2_3": run_additional_questions_2_3,
    }
    if survey_name not in survey_functions:
        raise ValueError(f"Unknown survey: {survey_name}.")
    return survey_functions[survey_name](client, run_number)


def save_results_to_excel(results, filename="survey_results.xlsx"):
    if not results:
        print("No results to save.")
        return
    long_df = pd.DataFrame(results)
    wide_df = long_df.pivot(index='run_number', columns='question', values='response')
    final_df = wide_df.reset_index()
    final_df.to_excel(filename, index=False)
    print(f"\n✅ Results successfully saved in wide format to {filename}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run LLM sustainability surveys (culture-free).")
    parser.add_argument("--model", required=True, choices=[m.value for m in ModelType], help="The LLM to use.")
    parser.add_argument("--survey", required=True,
                        choices=["AISPI", "SDG17", "SDG18", "SDG19", "AQ1", "AQ2_3"],
                        help="The survey to run.")
    parser.add_argument("--runs", type=int, default=1, help="Number of times to repeat the entire survey.")

    args = parser.parse_args()
    sanitized_model_name = args.model.replace('/', '_')
    output_filename = f"{args.survey}_{sanitized_model_name}.xlsx"

    try:
        print("=" * 60)
        print("🚀 INITIALIZING SURVEY SESSION")
        print(f"  - Model:         {args.model}")
        print(f"  - Survey:        {args.survey}")
        print(f"  - Total Runs:    {args.runs}")
        print(f"  - Output File:   {output_filename}")
        print("=" * 60)

        selected_model = ModelType(args.model)
        all_results = []
        client = LLMClient(selected_model)

        for i in range(1, args.runs + 1):
            print(f"\n--- Starting Run {i} of {args.runs} ---")
            single_run_results = run_survey(client, args.survey, run_number=i)
            all_results.extend(single_run_results)
            print(f"--- Finished Run {i} of {args.runs} ---")

        save_results_to_excel(all_results, filename=output_filename)
        print("\n🎉 Session complete.")
    except (ValueError, KeyError) as e:
        print(f"\n❌ Error: {e}")
    except Exception as e:
        print(f"\n❌ An unexpected error occurred: {e}")
