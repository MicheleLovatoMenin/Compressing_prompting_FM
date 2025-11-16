# 🚀 Compressing Prompting for Efficiency (Compressing Prompting FM)

This research project examines the balance between **human linguistic fluency** and **LLM inference efficiency**. The aim is to show that it is possible to **drastically reduce prompt length and cost** while maintaining *near-identical accuracy*, using a **Rule-Based Prompt Compression** methodology.

---

## 💡 Project Concept: Compressing Prompting

### The Problem
Advanced prompting techniques (e.g., *Chain-of-Thought*, *Few-Shot*) and naturally verbose human language significantly increase **Token Cost**.  
Most LLM APIs bill proportionally to the number of tokens, yet a substantial portion of tokens—articles, connectors, politeness markers, redundant adjectives—carry **low informational value** for the model.

### The Hypothesis
The essential information needed for reasoning and task completion lies in **core semantic units** rather than in the surrounding linguistic boilerplate.  
By applying systematic rules to remove or shorten:
- articles  
- adverbs  
- weak conjunctions  
- redundant modifiers  
- filler phrases  

…it is possible to achieve **large token savings** with **minimal accuracy loss**.

### The Objective
Validate the hypothesis via a controlled two-prompt comparison:

1. **Baseline (Original Prompt)**  
   The raw question/task written in standard natural language.

2. **Target (Compressed Prompt)**  
   A compressed version generated through rule-based linguistic reduction.

The project evaluates differences in:
- response accuracy  
- coherence  
- factual alignment  
- token cost  

to quantify the benefits of **Compressing Prompting**.

---

## Installation and Setup

Follow the steps below to correctly install and run the project.

### 1. Create and activate a virtual environment (recommended)

```bash
python3 -m venv .venv
source .venv/bin/activate      # macOS / Linux
# .venv\Scripts\activate       # Windows
```

### 2. Install project dependencies

```bash
pip install -r requirements.txt
```

### 3. Install the SpaCy model

SpaCy models must be installed separately from the requirements file.

```bash
python -m spacy download en_core_web_sm
```




