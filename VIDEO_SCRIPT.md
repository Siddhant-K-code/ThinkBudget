# ThinkBudget — Video Script

Target: 3-4 minutes. Twitter/YouTube short format.

---

## Setup (before recording)

Terminal 1 — start demo server:
```bash
thinkbudget demo --auto-traffic --traffic-interval 4
```

Terminal 2 — ready for demo commands:
```bash
# Have demo.sh ready, or type commands manually
```

Browser — open dashboard:
```
http://localhost:9100/dashboard
```

---

## Shot List

### Shot 1 — Hook (0:00–0:15)

**Screen:** Terminal, dark theme.

**Type or voiceover:**
> "Your reasoning model burns 4,000 thinking tokens on 'Hello'.
> Here's how to fix that."

**Action:** Show a curl to a raw vLLM endpoint, highlight the token count in the response.

---

### Shot 2 — The Problem (0:15–0:40)

**Screen:** Split — terminal left, simple diagram right.

**Voiceover/text:**
> "Reasoning models like DeepSeek-R1 wrap their thinking in <think> tags.
> Every query — trivial or complex — gets the same treatment.
> No budget. No visibility. Just GPU burn."

**Action:** Show two queries side by side:
- "Hello" → 3,800 thinking tokens, $0.0004
- "Debug this race condition" → 4,200 thinking tokens, $0.0005

Point: they cost almost the same. That's the problem.

---

### Shot 3 — ThinkBudget Classify (0:40–1:20)

**Screen:** Terminal, full screen.

**Action:** Run classifier on 5 queries, one by one:

```bash
thinkbudget classify "Hello"
# → TRIVIAL, 32 tokens

thinkbudget classify "What is the capital of France?"
# → SIMPLE, 128 tokens

thinkbudget classify "Compare REST vs GraphQL trade-offs"
# → MODERATE, 512 tokens

thinkbudget classify "Debug this race condition in Go"
# → COMPLEX, 2,048 tokens

thinkbudget classify "Prove the halting problem is undecidable"
# → COMPLEX, 2,048 tokens
```

**Voiceover:**
> "ThinkBudget classifies every query in under a millisecond.
> No LLM call. Pure heuristics. Five tiers, five budgets."

---

### Shot 4 — Proxy in Action (1:20–2:00)

**Screen:** Terminal.

**Action:** Send queries through the proxy, show the response headers:

```bash
curl -s -i http://localhost:9100/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"demo","messages":[{"role":"user","content":"Hello"}]}'
```

**Highlight the headers:**
```
X-ThinkBudget-Tier: trivial
X-ThinkBudget-Budget: 32
X-ThinkBudget-Thinking-Tokens: 3
X-ThinkBudget-Cost: $0.00000200
```

Then send a complex query, show the difference:
```
X-ThinkBudget-Tier: complex
X-ThinkBudget-Budget: 2048
X-ThinkBudget-Thinking-Tokens: 1847
X-ThinkBudget-Cost: $0.00018200
```

**Voiceover:**
> "Drop-in OpenAI-compatible proxy. Every response tells you
> the tier, the budget, the actual tokens used, and the GPU cost."

---

### Shot 5 — Dashboard (2:00–2:45)

**Screen:** Browser, dashboard full screen.

**Action:** Show the dashboard with queries streaming in (auto-traffic is running).

Walk through:
1. **Stats cards** — total queries, GPU cost, tokens saved, avg latency
2. **Tier distribution** — bar chart showing the mix
3. **GPU panel** — power, utilization, memory (simulated but looks real)
4. **Query table** — scroll through, point out tier badges, utilization bars, cost per query

**Voiceover:**
> "The dashboard shows everything in real-time.
> Every query, its tier, budget utilization, GPU cost.
> You can see exactly where your money goes."

---

### Shot 6 — The Savings (2:45–3:15)

**Screen:** Terminal, show stats endpoint.

```bash
curl -s http://localhost:9100/api/stats | python3 -m json.tool
```

**Highlight:**
- `tokens_saved: 245,000+`
- `total_cost` vs what it would have been

**Voiceover:**
> "In this demo, ThinkBudget saved over 245,000 thinking tokens.
> On a real RTX 4090 at $0.39/hour, that's real money.
> Simple queries get simple budgets. Complex queries get what they need."

---

### Shot 7 — Install & Close (3:15–3:40)

**Screen:** Terminal.

```bash
uv tool install thinkbudget
thinkbudget serve --backend-url http://localhost:8000
```

**Voiceover:**
> "Two commands. Point it at your vLLM, SGLang, or Ollama backend.
> Open source. Works on any GPU."

**End card:**
- GitHub: github.com/Siddhant-K-code/ThinkBudget
- PyPI: `uv tool install thinkbudget`

---

## Recording Tips

- Use a dark terminal theme (Dracula, One Dark, or similar)
- Font size 16-18pt so it's readable on mobile
- Dashboard looks best at 1920x1080 or 1440p
- Let auto-traffic run for ~30 seconds before showing the dashboard so it has data
- For the classify shots, pause between each to let the table render fully
- The demo.sh script handles pacing with press-enter prompts
