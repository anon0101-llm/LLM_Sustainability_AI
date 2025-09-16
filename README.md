# 🌍 Sustainability and AI through the Lens of LLMs

This repository contains **Python scripts** and **data** for comparing how different Large Language Models (LLMs) respond to:  
1. **Likert-scale surveys** on AI, sustainability, and their relationship  
2. **Budget allocation tasks** between sustainability and AI initiatives  

It supports both **API-based models** (OpenAI, Anthropic, DeepSeek) and **local Hugging Face models** (Mistral, LLaMA).  

---

## 📂 Repository Structure
```
├── data/
│   ├── AISPI_ChatGPT.xlsx
│   ├── SDG17_Mistral.xlsx
│   ├── Additional_Questions_DeepSeek.xlsx
│   └── ...
├── scripts/
│   └── survey_runner.py
├── requirements.txt
└── README.md
```

- `data/` → Collected results (Excel format)  
- `scripts/` → Survey runner & budget allocation scripts 
- `requirements.txt` → Python dependencies  
- `README.md` → Project documentation  

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/llm-sustainability-surveys.git
cd llm-sustainability-surveys
```

### 2. Install dependencies
We recommend Python **3.10+**.  
```bash
pip install -r requirements.txt
```

### 3. Configure API Keys
Create a `.env` file in the project root:

```
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
DEEPSEEK_API_KEY=your_deepseek_key
```

> Local models (Mistral, LLaMA) do not require API keys but need GPUs with sufficient memory.  

---

## 📊 Running Surveys

### Available Surveys
- **AISPI** – AI and Sustainability Perception Index (perceptions of AI as supporting vs. competing with sustainability)  
- **SDG17** – Expected impact of AI on the 17 UN Sustainable Development Goals  
- **SDG18** – Comparative importance of AI vs. Sustainability (priority rating)  
- **SDG19** – Future integration of AI and Sustainability  
- **AQ1** – Assessment of whether governments, industries, and organizations are doing enough  
- **AQ2_3** – Responsibility attribution and confidence in different actors (universities, international research organizations, technology companies, governments, NGOs)  

### Example Commands
Run AISPI with GPT-4o for 2 runs:
```bash
python scripts/run_survey.py --model gpt-4o --survey AISPI --runs 2
```

Run SDG17 with local LLaMA-3.3-70B:
```bash
python scripts/run_survey.py --model meta-llama/Llama-3.3-70B-Instruct --survey SDG17 --runs 1
```

Run Additional Questions with Mistral-Large:
```bash
python scripts/run_survey.py --model mistralai/Mistral-Large-Instruct-2407 --survey AQ2_3
```

Results will be saved as Excel files, e.g.:
```
AISPI_gpt-4o.xlsx
SDG17_Llama3.3-70B.xlsx
AQ2_3_Mistral-Large.xlsx
```

---

## 💰 Running Budget Allocation

In addition to surveys, the repository includes a **budget allocation experiment** where LLMs must divide $1,000,000 between a **sustainability initiative** and an **AI advertising initiative**.

### Example Command
```bash
python scripts/budget_allocation.py --model gpt-4o --runs 50
```

This produces an Excel file such as:
```
Budget_gpt-4o.xlsx
```

Each row contains the sustainability/AI allocation for a single iteration.

---

## 📑 Citation
If you use this repository, please cite it as:

```
@misc{llm_sustainability_2025,
  title   = {Choosing a Model, Shaping a Future: Comparing LLM Perspectives on Sustainability and its Relationship with AI},
  author  = {Bush, Annika and Aksoy, Meltem and Pauly, Markus and Ontrup, Greta},
  journal = {arXiv preprint arXiv:2505.14435},
  year    = {2025},
  url     = {https://arxiv.org/abs/2505.14435}
}

```

---

## 📜 License
This project is licensed under the MIT License.  

---

## 🙌 Acknowledgements
Developed at the Research Center Trustworthy Data Science and Security (RC-Trust).
