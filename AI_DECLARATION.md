# AI Assistance Declaration

## Tool

- **Tool:** GitHub Copilot using Copilot SDK in VS Code
- **Purpose:** Project implementation, code generation, data auditing, documentation, testing support, and dashboard development

## Disclosure

AI-assisted contributions remain subject to the project owner's human review and academic disclosure obligations.

| File or Area | AI-Assisted Contribution | Human Modification / Verification |
|---|---|---|
| `src/data_processing.py` | Proposed cleaning, normalization, validation, deduplication, and age feature code | Dataset audit and cleaning policy reviewed; no invalid-value or duplicate results were fabricated |
| `notebooks/01_data_exploration.ipynb` | Generated EDA structure and charts | Chart execution and interpretation require human review |
| `src/train.py` | Proposed preprocessing pipeline, split, baseline, and ensemble definitions | Split metadata and model configuration reviewed |
| `src/evaluate.py` | Implemented metric and timing helpers | Metric units and formulas verified |
| `artifacts/*.csv` / `artifacts/*.json` | Generated from actual executed model and data operations | Contents require human final review against console and notebook results |
| `app.py` | Generated six-page Streamlit dashboard | Human verification of prediction inputs and displayed conversion required |
| Tests | Proposed preprocessing and model roundtrip tests | Executed by pytest; 4 tests passed |
| README and this declaration | Generated project and AI disclosure documentation | Repository owner must verify and maintain personal observations and references |

## Academic Integrity

The student must be able to explain each model, preprocessing decision, metric, and result, and must disclose AI assistance according to university rules. AI-generated text or code must not be represented as independently written without review.
