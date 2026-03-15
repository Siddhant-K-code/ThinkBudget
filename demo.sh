#!/bin/bash
# ThinkBudget Demo Script
# Run this in a terminal while recording for a video.
# Start the demo server first: thinkbudget demo --auto-traffic
#
# Usage: bash demo.sh

set -euo pipefail

PORT="${PORT:-9100}"
BASE="http://localhost:${PORT}"

# Colors
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
DIM='\033[2m'
BOLD='\033[1m'
NC='\033[0m'

pause() {
  echo ""
  read -p "$(echo -e "${DIM}[press enter]${NC}")" _
}

echo -e "${BOLD}╔══════════════════════════════════════════════════════════╗${NC}"
echo -e "${BOLD}║              ThinkBudget — Live Demo                     ║${NC}"
echo -e "${BOLD}╚══════════════════════════════════════════════════════════╝${NC}"
echo ""

# ── 1. Classify queries ──
echo -e "${CYAN}━━━ 1. Query Classification (no LLM call, <1ms) ━━━${NC}"
pause

echo -e "${YELLOW}$ thinkbudget classify \"Hello\"${NC}"
thinkbudget classify "Hello"
pause

echo -e "${YELLOW}$ thinkbudget classify \"What is the capital of France?\"${NC}"
thinkbudget classify "What is the capital of France?"
pause

echo -e "${YELLOW}$ thinkbudget classify \"Compare REST vs GraphQL trade-offs for mobile\"${NC}"
thinkbudget classify "Compare REST vs GraphQL trade-offs for mobile"
pause

echo -e "${YELLOW}$ thinkbudget classify \"Debug this race condition in Go with goroutines writing to a shared map\"${NC}"
thinkbudget classify "Debug this race condition in Go with goroutines writing to a shared map"
pause

echo -e "${YELLOW}$ thinkbudget classify \"Prove the halting problem is undecidable using diagonalization\"${NC}"
thinkbudget classify "Prove the halting problem is undecidable using diagonalization"
pause

# ── 2. Send queries through the proxy ──
echo -e "${CYAN}━━━ 2. Proxy — Budget-Controlled Inference ━━━${NC}"
echo ""
echo -e "${DIM}Sending a TRIVIAL query (budget: 32 tokens)...${NC}"
pause

echo -e "${YELLOW}$ curl -s ${BASE}/v1/chat/completions -d '{\"model\":\"demo\",\"messages\":[{\"role\":\"user\",\"content\":\"Hello\"}]}' -H 'Content-Type: application/json'${NC}"
echo ""
RESP=$(curl -s -i "${BASE}/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{"model":"demo","messages":[{"role":"user","content":"Hello"}]}')

# Show headers
echo -e "${GREEN}Response Headers:${NC}"
echo "$RESP" | grep -i "X-ThinkBudget" | while read line; do
  echo -e "  ${GREEN}${line}${NC}"
done
echo ""

# Show body
echo -e "${GREEN}Response:${NC}"
echo "$RESP" | tail -1 | python3 -m json.tool 2>/dev/null || echo "$RESP" | tail -1
pause

echo -e "${DIM}Now sending a COMPLEX query (budget: 2048 tokens)...${NC}"
pause

echo -e "${YELLOW}$ curl -s ${BASE}/v1/chat/completions -d '{\"model\":\"demo\",\"messages\":[{\"role\":\"user\",\"content\":\"Design a distributed task queue with at-least-once delivery\"}]}'${NC}"
echo ""
RESP=$(curl -s -i "${BASE}/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{"model":"demo","messages":[{"role":"user","content":"Design a distributed task queue with at-least-once delivery, priorities, and dead letter queues. Walk through the architecture step by step."}]}')

echo -e "${GREEN}Response Headers:${NC}"
echo "$RESP" | grep -i "X-ThinkBudget" | while read line; do
  echo -e "  ${GREEN}${line}${NC}"
done
echo ""
echo -e "${GREEN}Response (truncated):${NC}"
echo "$RESP" | tail -1 | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['choices'][0]['message']['content'][:300] + '...')" 2>/dev/null
pause

# ── 3. Stats ──
echo -e "${CYAN}━━━ 3. Cost Tracking ━━━${NC}"
pause

echo -e "${YELLOW}$ curl -s ${BASE}/api/stats${NC}"
curl -s "${BASE}/api/stats" | python3 -c "
import sys, json
d = json.load(sys.stdin)
print(f\"  Total queries:    {d['total_queries']}\")
print(f\"  Total GPU cost:   \${d['total_cost']:.6f}\")
print(f\"  Tokens saved:     {d['total_tokens_saved']:,}\")
print(f\"  Cost saved:       \${d['total_cost_saved']:.4f}\")
print(f\"  Avg latency:      {d['avg_latency_ms']:.0f}ms\")
print(f\"  GPU:              {d['gpu_info']['name']}\")
print(f\"  Tier distribution:\")
for tier, count in d['tier_distribution'].items():
    print(f\"    {tier:>10}: {count}\")
"
pause

echo -e "${CYAN}━━━ 4. Open the Dashboard ━━━${NC}"
echo ""
echo -e "  ${BOLD}→ http://localhost:${PORT}/dashboard${NC}"
echo ""
echo -e "${DIM}The dashboard shows live query stream, tier distribution,${NC}"
echo -e "${DIM}GPU metrics, and cumulative cost savings.${NC}"
pause

echo -e "${GREEN}${BOLD}Demo complete.${NC}"
