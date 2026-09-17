#!/usr/bin/env bash
# Downloads machine-readable documentation for the libraries in our tech stack into
# docs/libraries/ (gitignored). Re-runnable: overwrites what it fetched before.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$ROOT/docs/libraries"
mkdir -p "$OUT"

fetch() {
  # fetch <dest-relative-path> <url>
  local dest="$OUT/$1" url="$2" code size
  mkdir -p "$(dirname "$dest")"
  code=$(curl -sS -L --max-time 120 -o "$dest.tmp" -w '%{http_code}' "$url" || echo 000)
  size=$(wc -c <"$dest.tmp" 2>/dev/null | tr -d ' ')
  if [ "$code" = "200" ] && [ "${size:-0}" -gt 500 ]; then
    mv "$dest.tmp" "$dest"
    printf 'OK    %7s bytes  %s\n' "$size" "$1"
  else
    rm -f "$dest.tmp"
    printf 'MISS  http %s        %s  (%s)\n' "$code" "$1" "$url"
  fi
}

echo "== LlamaIndex =="
fetch llamaindex/llms.txt https://developers.llamaindex.ai/llms.txt
# Their docs expose raw markdown by appending index.md to any page URL (stated in llms.txt).
fetch llamaindex/bm25_retriever.md https://developers.llamaindex.ai/python/framework/integrations/retrievers/bm25_retriever/index.md
fetch llamaindex/rrf_fusion.md https://developers.llamaindex.ai/python/examples/retrievers/reciprocal_rerank_fusion/index.md
fetch llamaindex/agents.md https://developers.llamaindex.ai/python/framework/module_guides/deploying/agents/index.md
fetch llamaindex/agent_tutorial.md https://developers.llamaindex.ai/python/framework/understanding/agent/index.md
fetch llamaindex/node_parsers.md https://developers.llamaindex.ai/python/framework/module_guides/loading/node_parsers/index.md
fetch llamaindex/node_postprocessors.md https://developers.llamaindex.ai/python/framework/module_guides/querying/node_postprocessors/index.md
fetch llamaindex/vector_stores.md https://developers.llamaindex.ai/python/framework/module_guides/storing/vector_stores/index.md
fetch llamaindex/retrievers.md https://developers.llamaindex.ai/python/framework/module_guides/querying/retriever/index.md
fetch llamaindex/metadata_filtering.md https://developers.llamaindex.ai/python/examples/vector_stores/chroma_metadata_filter/index.md
fetch llamaindex/voyage_embeddings.md https://developers.llamaindex.ai/python/examples/embeddings/voyageai/index.md
fetch llamaindex/voyage_rerank.md https://developers.llamaindex.ai/python/examples/node_postprocessor/voyageairerank/index.md

echo "== Chroma =="
fetch chroma/llms.txt https://docs.trychroma.com/llms.txt
fetch chroma/llms-full.txt https://docs.trychroma.com/llms-full.txt

echo "== Voyage AI =="
fetch voyage/llms.txt https://docs.voyageai.com/llms.txt
for p in introduction api-key-and-installation quickstart-tutorial embeddings reranker \
         tokenization flexible-dimensions-and-quantization batch-inference error-codes \
         rate-limits pricing; do
  fetch "voyage/$p.md" "https://docs.voyageai.com/docs/$p.md"
done

echo "== Azure Document Intelligence =="
fetch azure-di/layout-model.md https://raw.githubusercontent.com/MicrosoftDocs/azure-ai-docs/main/articles/ai-services/document-intelligence/prebuilt/layout.md
fetch azure-di/sdk-python-quickstart.md https://raw.githubusercontent.com/MicrosoftDocs/azure-ai-docs/main/articles/ai-services/document-intelligence/quickstarts/get-started-sdks-rest-api.md
fetch azure-di/markdown-output.md https://raw.githubusercontent.com/MicrosoftDocs/azure-ai-docs/main/articles/ai-services/document-intelligence/concept/markdown-elements.md
fetch azure-di/service-limits.md https://raw.githubusercontent.com/MicrosoftDocs/azure-ai-docs/main/articles/ai-services/document-intelligence/service-limits.md

echo "== OpenRouter (generation) =="
fetch openrouter/llms.txt https://openrouter.ai/docs/llms.txt
fetch openrouter/llms-full.txt https://openrouter.ai/docs/llms-full.txt

echo "== bm25s =="
fetch bm25s/README.md https://raw.githubusercontent.com/xhluca/bm25s/main/README.md

echo
echo "Wrote into: $OUT"
find "$OUT" -type f | sort
