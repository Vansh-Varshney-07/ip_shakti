#!/usr/bin/env bash
set -euo pipefail

# Run the reproducible prompt corpus against the local authenticated API.
# Usage: BASE_URL=http://127.0.0.1:8000 bash scripts/run_prompt_suite.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROMPT_FILE="${1:-${ROOT_DIR}/docs/prompt_suite_30.txt}"
BASE_URL="${BASE_URL:-http://127.0.0.1:8000}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/artifacts/prompt-runs}"
CURL_MAX_TIME="${CURL_MAX_TIME:-180}"
PYTHON_BIN="${PYTHON_BIN:-python}"

if [[ ! -f "${PROMPT_FILE}" ]]; then
  echo "Prompt file not found: ${PROMPT_FILE}" >&2
  exit 1
fi
command -v curl >/dev/null || { echo "curl is required" >&2; exit 1; }
command -v "${PYTHON_BIN}" >/dev/null || { echo "Python is required: ${PYTHON_BIN}" >&2; exit 1; }

python_path() {
  if command -v wslpath >/dev/null 2>&1; then
    wslpath -w "$1"
  else
    printf '%s' "$1"
  fi
}

mkdir -p "${OUTPUT_DIR}"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
RAW_FILE="${OUTPUT_DIR}/prompt-suite-${RUN_ID}.jsonl"
SUMMARY_FILE="${OUTPUT_DIR}/prompt-suite-${RUN_ID}.summary.json"
LOG_FILE="${OUTPUT_DIR}/prompt-suite-${RUN_ID}.log"
TMP_DIR="${OUTPUT_DIR}/.tmp-${RUN_ID}"
mkdir -p "${TMP_DIR}"
trap 'rm -rf "${TMP_DIR}"' EXIT

echo "[$(date -u +%FT%TZ)] authenticating against ${BASE_URL}" | tee "${LOG_FILE}"
TOKEN_JSON="$(curl -fsS --max-time "${CURL_MAX_TIME}" -X POST "${BASE_URL}/auth/token" \
  -H 'Content-Type: application/json' \
  --data '{"username":"prompt-suite","password":"prompt-suite","scope":"api ingest:write"}')"
TOKEN="$(printf '%s' "${TOKEN_JSON}" | "${PYTHON_BIN}" -c 'import json,sys; print(json.load(sys.stdin)["access_token"])' | tr -d '\r')"

: > "${RAW_FILE}"
TOTAL=0
exec 3< "${PROMPT_FILE}"
while IFS=$'\t' read -r prompt_id persona jurisdiction language prompt <&3; do
  [[ -z "${prompt_id}" || "${prompt_id}" == \#* ]] && continue
  TOTAL=$((TOTAL + 1))
  REQUEST_FILE="${TMP_DIR}/${TOTAL}.request.json"
  RESPONSE_FILE="${TMP_DIR}/${TOTAL}.response.json"
  "${PYTHON_BIN}" - "$(python_path "${REQUEST_FILE}")" "${prompt}" "${language}" "${jurisdiction}" <<'PY'
import json
import sys

path, prompt, language, jurisdiction = sys.argv[1:]
payload = {
    "query": prompt,
    "user_id": "prompt-suite",
    "language": language,
    "jurisdiction": jurisdiction,
    "require_citations": True,
    "max_results": 10,
}
with open(path, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, ensure_ascii=False)
PY

  START_NS="$("${PYTHON_BIN}" -c 'import time; print(time.time_ns())' | tr -d '\r')"
  HTTP_CODE="$(curl -sS --max-time "${CURL_MAX_TIME}" -o "${RESPONSE_FILE}" -w '%{http_code}' \
    -X POST "${BASE_URL}/query" \
    -H "Authorization: Bearer ${TOKEN}" \
    -H 'Content-Type: application/json' \
    --data-binary "@${REQUEST_FILE}" < /dev/null || true)"
  END_NS="$("${PYTHON_BIN}" -c 'import time; print(time.time_ns())' | tr -d '\r')"

  "${PYTHON_BIN}" - "$(python_path "${RESPONSE_FILE}")" "$(python_path "${RAW_FILE}")" "${prompt_id}" "${persona}" "${jurisdiction}" "${language}" "${HTTP_CODE}" "${START_NS}" "${END_NS}" <<'PY'
import json
import sys

response_path, output_path, prompt_id, persona, jurisdiction, language, http_code, start_ns, end_ns = sys.argv[1:]
try:
    with open(response_path, encoding="utf-8") as handle:
        payload = json.load(handle)
except Exception as exc:
    payload = {"success": False, "error": f"invalid_json_response: {exc}"}

data = payload.get("data") or {}
answer = data.get("answer") or {}
stats = answer.get("retrieval_stats") or {}
record = {
    "prompt_id": prompt_id,
    "persona": persona,
    "jurisdiction_requested": jurisdiction,
    "language_requested": language,
    "http_status": int(http_code or 0),
    "elapsed_ms": round((int(end_ns) - int(start_ns)) / 1_000_000, 2),
    "success": bool(payload.get("success", False)),
    "query_id": data.get("query_id"),
    "intent": data.get("intent"),
    "jurisdiction_returned": data.get("jurisdiction"),
    "retrieved_count": len(data.get("retrieved_chunks") or []),
    "citation_count": len(answer.get("citations") or []),
    "answer_text": "\n".join(segment.get("text", "") for segment in answer.get("segments") or []),
    "citation_sources": [
        (citation.get("source_reference") or {}).get("source_name")
        for citation in answer.get("citations") or []
    ],
    "confidence": answer.get("overall_confidence", 0),
    "positive_semantic_searches": stats.get("positive_semantic_searches", 0),
    "negative_semantic_searches": stats.get("negative_semantic_searches", 0),
    "abstained": bool(not (data.get("retrieved_chunks") or [])),
    "warnings": data.get("warnings") or [],
    "error": payload.get("error") or payload.get("detail"),
}
with open(output_path, "a", encoding="utf-8") as handle:
    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
print(
    f"{prompt_id}: http={record['http_status']} intent={record['intent']} "
    f"retrieved={record['retrieved_count']} citations={record['citation_count']} "
    f"elapsed_ms={record['elapsed_ms']}"
)
PY
done >> "${LOG_FILE}"
exec 3<&-
cat "${LOG_FILE}"

"${PYTHON_BIN}" - "$(python_path "${RAW_FILE}")" "$(python_path "${SUMMARY_FILE}")" "$(python_path "${PROMPT_FILE}")" <<'PY'
import json
import sys
from collections import Counter

raw_path, summary_path, prompt_path = sys.argv[1:]
with open(raw_path, encoding="utf-8") as handle:
    records = [json.loads(line) for line in handle if line.strip()]

summary = {
    "prompt_file": prompt_path,
    "total": len(records),
    "http_2xx": sum(item["http_status"] == 200 for item in records),
    "successful": sum(item["success"] for item in records),
    "with_citations": sum(item["citation_count"] > 0 for item in records),
    "abstentions": sum(item["abstained"] for item in records),
    "warnings": sum(bool(item["warnings"]) for item in records),
    "average_elapsed_ms": round(sum(item["elapsed_ms"] for item in records) / max(len(records), 1), 2),
    "intents": dict(Counter(item["intent"] or "unknown" for item in records)),
    "jurisdictions": dict(Counter(item["jurisdiction_returned"] or "unknown" for item in records)),
    "positive_search_counts": dict(Counter(str(item["positive_semantic_searches"]) for item in records)),
    "negative_search_counts": dict(Counter(str(item["negative_semantic_searches"]) for item in records)),
    "failed_prompt_ids": [
        item["prompt_id"] for item in records
        if not item["success"] or item["http_status"] != 200
    ],
    "zero_evidence_prompt_ids": [
        item["prompt_id"] for item in records if item["abstained"]
    ],
}
with open(summary_path, "w", encoding="utf-8") as handle:
    json.dump(summary, handle, indent=2, ensure_ascii=False)
print(json.dumps(summary, indent=2, ensure_ascii=False))
PY

echo "Raw records: ${RAW_FILE}"
echo "Summary: ${SUMMARY_FILE}"
