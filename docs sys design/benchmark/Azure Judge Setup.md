# Azure DeepSeek judge setup

This guide configures the Azure-hosted `DeepSeek-V4-Flash` deployment used by
the FinanceBench binary judge.

## 1. Verify the deployed offering

In [Microsoft Foundry](https://ai.azure.com/), open the deployment and confirm:

- the publisher is **DeepSeek**;
- the offering says **Direct from Azure**, not `FW-DeepSeek-V4-Flash` from
  Fireworks;
- provisioning status is **Succeeded**;
- the deployment name is copied exactly for `configs/financebench.toml`.

Direct-from-Azure models are eligible for Microsoft for Startups credits;
third-party Marketplace billing may not be. See
[startup-credit coverage](https://learn.microsoft.com/en-us/startups/benefits/technical-benefits/azure-credits/foundry-model-sponsorship-coverage)
and the [DeepSeek-V4-Flash model card](https://ai.azure.com/catalog/models/DeepSeek-V4-Flash).

## 2. Test the deployment in Foundry

Open the deployment playground and send `Return the number 1.` This verifies the
deployment before testing this repository.

## 3. Store the connection values

The project endpoint has this form:

```text
https://RESOURCE.services.ai.azure.com/api/projects/PROJECT
```

Store it unchanged in the ignored `.env`. The Python client will append
`/openai/v1` when it constructs the Chat Completions base URL:

```dotenv
AZURE_DEEPSEEK_ENDPOINT=https://RESOURCE.services.ai.azure.com/api/projects/PROJECT
AZURE_DEEPSEEK_API_KEY="your-key-here"
```

Quotes around the key are valid when the file is loaded by the shell. Do not
commit `.env`, paste the key into documentation, or send it in chat.
`.env.example` contains only blank variable names and is safe to commit.

Load the variables in every new terminal session:

```bash
cd "/Users/zubairasim/Documents/SEC RAG/.worktrees/azure-ragas-judge"
set -a
source .env
set +a
```

## 4. Understand the two unrelated installations

The `az` command belongs to the system-level **Azure CLI**. It is not a Python
package and does not belong in `pyproject.toml`. On macOS with Homebrew:

```bash
brew install azure-cli
az version
az login
az account show --query "{subscription:name, tenant:tenantId}" --output table
```

Installing Azure CLI does not add a dependency to this Python project. Use it
only when you want to inspect or administer Azure from the terminal; the portal
can perform the same setup tasks.

Do **not** run `pip install --upgrade openai azure-identity` for this project:

- `openai` is already pinned by `uv` in `pyproject.toml` and `uv.lock`;
- installing it with `pip --upgrade` can make the environment disagree with the
  lockfile;
- `azure-identity` is for Microsoft Entra login, while this first version uses
  the API key already stored in `.env`.

If Entra authentication is adopted later, add a reviewed exact version with
`uv add "azure-identity==VERSION"`, which updates both project files.

## 5. How the code will call DeepSeek

The existing GLM generator remains on OpenRouter's Responses API. Only the
judge uses Azure DeepSeek and Chat Completions:

```text
OpenRouter GLM generation → client.responses.create()
Azure DeepSeek judging    → client.chat.completions.create()
```

Microsoft's current DeepSeek documentation uses a Foundry project endpoint,
appends `/openai/v1`, and passes the deployment name as `model`:
[use reasoning models with Foundry](https://learn.microsoft.com/en-us/azure/foundry/foundry-models/how-to/use-chat-reasoning).

## 6. Throttling behaviour

The judge will send requests sequentially rather than concurrently. Its OpenAI
client will use `max_retries=5`; the SDK automatically retries HTTP 429 rate
limits and transient server/network failures with backoff. Completed judgments
are saved immediately, so resuming skips them and retries only unfinished jobs.

Azure explains its TPM/RPM limits and retry headers here:
[manage Foundry model quota](https://learn.microsoft.com/en-us/azure/foundry/openai/how-to/quota).

## 7. Run one paid smoke test

After implementation and fake-client tests pass, judge a results directory that
contains one successful generated answer:

```bash
uv run sec-rag-benchmark judge \
  --config configs/financebench.toml \
  --run-dir results/YOUR-ONE-ANSWER-RUN
```

This makes two paid calls because the answer is checked in both prompt orders.
Open `judgments.jsonl` and inspect both verdicts, combined accuracy, returned
model, usage and latency before judging a larger run.

## Troubleshooting

- **401/403:** confirm the key belongs to this Foundry resource and was loaded
  into the terminal.
- **404:** confirm the TOML deployment name exactly matches Foundry and the
  project endpoint has not been replaced with a deployment/playground URL.
- **429:** allow the configured retries to run; if failures persist, inspect the
  deployment's TPM/RPM allocation and resume later.
- **Unexpected bill:** stop and confirm the deployment is Direct from Azure and
  belongs to the startup-credit subscription.
