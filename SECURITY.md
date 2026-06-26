# Security Policy

## Reporting a Vulnerability
Please report security issues privately via a GitHub security advisory or by
emailing the maintainers. Do not open public issues for security reports.

## Secrets
This project uses the following credentials, loaded from environment variables
or Colab secrets — never hard-coded and never committed:

- `OPENAI_API_KEY` — GPT-4o-mini judge (training/evaluation only)
- `WANDB_API_KEY` — experiment tracking (optional)
- Hugging Face token — only required if the model repo is private

No credentials are stored in this repository. If you fork or clone this project,
supply your own keys through environment variables.
