"""Generate benchmark datasets for ThinkBudget evaluation.

Creates JSONL files with queries at each complexity tier.
These are used to measure cost savings and quality impact
of adaptive budget control.
"""

from __future__ import annotations

import json
from pathlib import Path

DATASETS = {
    "trivial": [
        {"query": "Hello", "expected_tier": "trivial"},
        {"query": "Hi there", "expected_tier": "trivial"},
        {"query": "Thanks", "expected_tier": "trivial"},
        {"query": "Yes", "expected_tier": "trivial"},
        {"query": "OK", "expected_tier": "trivial"},
        {"query": "What is 2+2?", "expected_tier": "trivial"},
        {"query": "What color is the sky?", "expected_tier": "trivial"},
        {"query": "Say hello", "expected_tier": "trivial"},
        {"query": "Ping", "expected_tier": "trivial"},
        {"query": "What day is it?", "expected_tier": "trivial"},
        {"query": "How are you?", "expected_tier": "trivial"},
        {"query": "Good morning", "expected_tier": "trivial"},
        {"query": "Bye", "expected_tier": "trivial"},
        {"query": "Got it", "expected_tier": "trivial"},
        {"query": "Sure", "expected_tier": "trivial"},
        {"query": "What is Python?", "expected_tier": "trivial"},
        {"query": "Name a fruit", "expected_tier": "trivial"},
        {"query": "What is HTTP?", "expected_tier": "trivial"},
        {"query": "What is JSON?", "expected_tier": "trivial"},
        {"query": "What is an API?", "expected_tier": "trivial"},
    ],
    "simple": [
        {"query": "What is the capital of France?", "expected_tier": "simple"},
        {"query": "Convert 100 Fahrenheit to Celsius", "expected_tier": "simple"},
        {"query": "What is a REST API?", "expected_tier": "simple"},
        {"query": "Define polymorphism in programming", "expected_tier": "simple"},
        {"query": "What is the difference between HTTP and HTTPS?", "expected_tier": "simple"},
        {"query": "How many bytes in a kilobyte?", "expected_tier": "simple"},
        {"query": "What is a Docker container?", "expected_tier": "simple"},
        {"query": "Translate 'hello world' to Spanish", "expected_tier": "simple"},
        {"query": "What is the population of Tokyo?", "expected_tier": "simple"},
        {"query": "What does CRUD stand for?", "expected_tier": "simple"},
        {"query": "What is a hash table?", "expected_tier": "simple"},
        {"query": "What is TCP/IP?", "expected_tier": "simple"},
        {"query": "What is the time complexity of binary search?", "expected_tier": "simple"},
        {"query": "What is a foreign key in SQL?", "expected_tier": "simple"},
        {"query": "What is the difference between git merge and rebase?", "expected_tier": "simple"},
        {"query": "What is a lambda function?", "expected_tier": "simple"},
        {"query": "What is CORS?", "expected_tier": "simple"},
        {"query": "What is a webhook?", "expected_tier": "simple"},
        {"query": "What is OAuth?", "expected_tier": "simple"},
        {"query": "What is a microservice?", "expected_tier": "simple"},
    ],
    "moderate": [
        {
            "query": "Compare REST vs GraphQL for a mobile app backend. What are the trade-offs?",
            "expected_tier": "moderate",
        },
        {
            "query": "Explain the CAP theorem and give an example of each trade-off",
            "expected_tier": "moderate",
        },
        {
            "query": "Write a Python function that finds the longest palindromic substring",
            "expected_tier": "moderate",
        },
        {
            "query": "What are the implications of using eventual consistency in a distributed system?",
            "expected_tier": "moderate",
        },
        {
            "query": "Explain how a B-tree index works in PostgreSQL and when to use it vs a hash index",
            "expected_tier": "moderate",
        },
        {
            "query": "Design a rate limiter using the token bucket algorithm. Show the implementation.",
            "expected_tier": "moderate",
        },
        {
            "query": "What are the trade-offs between monorepo and polyrepo for a team of 20 engineers?",
            "expected_tier": "moderate",
        },
        {
            "query": "Explain the difference between optimistic and pessimistic locking with examples",
            "expected_tier": "moderate",
        },
        {
            "query": "Write a TypeScript generic type that extracts all nested keys from an object type",
            "expected_tier": "moderate",
        },
        {
            "query": "Compare Kubernetes Deployments vs StatefulSets. When would you use each?",
            "expected_tier": "moderate",
        },
        {
            "query": "Explain how TLS 1.3 handshake works step by step",
            "expected_tier": "moderate",
        },
        {
            "query": "What is the difference between horizontal and vertical scaling? Give examples.",
            "expected_tier": "moderate",
        },
        {
            "query": "Write a SQL query to find the second highest salary in each department",
            "expected_tier": "moderate",
        },
        {
            "query": "Explain event sourcing and CQRS. When should you use them together?",
            "expected_tier": "moderate",
        },
        {
            "query": "Compare WebSockets, Server-Sent Events, and long polling for real-time features",
            "expected_tier": "moderate",
        },
        {
            "query": "Explain how garbage collection works in Go vs Java",
            "expected_tier": "moderate",
        },
        {
            "query": "Design a simple pub/sub system. What data structures would you use?",
            "expected_tier": "moderate",
        },
        {
            "query": "What are the security implications of storing JWTs in localStorage vs cookies?",
            "expected_tier": "moderate",
        },
        {
            "query": "Explain the reactor pattern and how Node.js uses it",
            "expected_tier": "moderate",
        },
        {
            "query": "Compare different load balancing algorithms: round robin, least connections, consistent hashing",
            "expected_tier": "moderate",
        },
    ],
    "complex": [
        {
            "query": "Debug this race condition: I have two goroutines writing to a shared map. Sometimes the program panics with 'concurrent map writes'. The map is protected by a sync.Mutex but the panic still occurs. Here's the code:\n```go\nvar mu sync.Mutex\nvar cache = make(map[string]int)\n\nfunc update(key string, val int) {\n    mu.Lock()\n    cache[key] = val\n    mu.Unlock()\n}\n\nfunc read(key string) int {\n    return cache[key] // sometimes panics\n}\n```\nWhat's wrong and how do I fix it?",
            "expected_tier": "complex",
        },
        {
            "query": "Design a distributed task queue that handles at-least-once delivery, supports priorities, delayed execution, and dead letter queues. Walk through the architecture step by step.",
            "expected_tier": "complex",
        },
        {
            "query": "Analyze the time and space complexity of this algorithm and suggest optimizations:\n```python\ndef find_all_paths(graph, start, end, path=[]):\n    path = path + [start]\n    if start == end:\n        return [path]\n    paths = []\n    for node in graph[start]:\n        if node not in path:\n            new_paths = find_all_paths(graph, node, end, path)\n            paths.extend(new_paths)\n    return paths\n```",
            "expected_tier": "complex",
        },
        {
            "query": "Design a multi-tenant SaaS authorization system using relationship-based access control (ReBAC). Support organizations, teams, projects, and resources with inheritance. Show the schema and query patterns.",
            "expected_tier": "complex",
        },
        {
            "query": "Implement a lock-free concurrent queue in C++ using compare-and-swap. Handle the ABA problem. Explain each design decision.",
            "expected_tier": "complex",
        },
        {
            "query": "Design a real-time collaborative text editor. Compare OT vs CRDT approaches. Which would you choose for a team of 100 concurrent editors and why? Show the data structure.",
            "expected_tier": "complex",
        },
        {
            "query": "I'm seeing intermittent 502 errors in production. The service is behind an nginx reverse proxy, running on Kubernetes with 3 replicas. The errors correlate with deployment rollouts but also happen randomly. Walk me through a systematic debugging approach.",
            "expected_tier": "complex",
        },
        {
            "query": "Design a search engine for a 10TB document corpus. Cover indexing, ranking, query processing, and scaling. What trade-offs would you make for a team of 5 engineers?",
            "expected_tier": "complex",
        },
        {
            "query": "Implement a Raft consensus algorithm in pseudocode. Cover leader election, log replication, and safety. What are the edge cases that most implementations get wrong?",
            "expected_tier": "complex",
        },
        {
            "query": "Design a feature flag system that supports gradual rollouts, A/B testing, user targeting, and real-time updates. The system needs to handle 100K requests/second with <5ms latency. Show the architecture.",
            "expected_tier": "complex",
        },
        {
            "query": "Analyze this memory leak in a React application. The component unmounts but the WebSocket connection stays open and keeps updating state. Show the fix and explain the React lifecycle implications.\n```jsx\nfunction LiveFeed({ url }) {\n  const [data, setData] = useState([]);\n  useEffect(() => {\n    const ws = new WebSocket(url);\n    ws.onmessage = (e) => setData(prev => [...prev, JSON.parse(e.data)]);\n  }, [url]);\n  return <div>{data.map(d => <Item key={d.id} {...d} />)}</div>;\n}\n```",
            "expected_tier": "complex",
        },
        {
            "query": "Design a CDC (Change Data Capture) pipeline from PostgreSQL to Elasticsearch. Handle schema evolution, exactly-once semantics, and backfill. What tools would you use and why?",
            "expected_tier": "complex",
        },
        {
            "query": "Implement a circuit breaker pattern with half-open state, configurable thresholds, and metrics. Show the state machine and thread-safety considerations.",
            "expected_tier": "complex",
        },
        {
            "query": "Design an API gateway that handles authentication, rate limiting, request transformation, and circuit breaking. Compare build vs buy. Show the request flow.",
            "expected_tier": "complex",
        },
        {
            "query": "Optimize this PostgreSQL query that takes 30 seconds on a 50M row table:\n```sql\nSELECT u.name, COUNT(o.id) as order_count, SUM(o.total) as total_spent\nFROM users u\nLEFT JOIN orders o ON u.id = o.user_id\nWHERE o.created_at > NOW() - INTERVAL '30 days'\nGROUP BY u.id, u.name\nHAVING SUM(o.total) > 1000\nORDER BY total_spent DESC\nLIMIT 100;\n```\nShow the EXPLAIN plan analysis and optimization steps.",
            "expected_tier": "complex",
        },
    ],
    "deep": [
        {
            "query": "Prove that the halting problem is undecidable using a diagonalization argument. Then explain the implications for static analysis of AI agent behavior — can we ever guarantee that an agent will terminate?",
            "expected_tier": "deep",
        },
        {
            "query": "Design a complete distributed database from scratch. Cover: storage engine (LSM-tree vs B-tree), replication (sync vs async, quorum), partitioning (range vs hash), transactions (2PC vs Saga), query planning, and failure recovery. For each component, explain the trade-offs and your design choice. Target: 1M TPS, 99.99% availability, strong consistency.",
            "expected_tier": "deep",
        },
        {
            "query": "Implement a transformer attention mechanism from scratch in PyTorch. Include multi-head attention, positional encoding, and the full forward pass. Then analyze the computational complexity and propose three optimizations (flash attention, sparse attention, linear attention). For each optimization, show the modified code and benchmark the memory/compute trade-offs.",
            "expected_tier": "deep",
        },
        {
            "query": "Design a formal verification system for smart contracts. Cover: the specification language, the verification algorithm, soundness guarantees, and limitations. Implement a simplified version that can verify basic properties (no reentrancy, bounded gas usage). Show how it handles the DAO hack pattern.",
            "expected_tier": "deep",
        },
        {
            "query": "Derive the backpropagation algorithm from first principles using the chain rule. Then extend it to handle: (1) batch normalization layers, (2) residual connections, (3) attention mechanisms. For each extension, show the gradient computation and explain numerical stability considerations.",
            "expected_tier": "deep",
        },
        {
            "query": "Design a capability-based security system for autonomous AI agents operating in a multi-tenant environment. The system must handle: capability delegation with attenuation, revocation in O(1), audit logging with tamper evidence, and cross-agent capability transfer. Prove that the system prevents privilege escalation. Show the formal model and implementation.",
            "expected_tier": "deep",
        },
        {
            "query": "Implement a garbage collector for a programming language runtime. Compare mark-and-sweep, copying, and generational approaches. Implement a generational GC with: nursery (copying), old generation (mark-compact), write barriers, and concurrent marking. Analyze pause times and throughput for different workload patterns.",
            "expected_tier": "deep",
        },
        {
            "query": "Design and implement a type inference engine for a Hindley-Milner type system. Support: let-polymorphism, type classes (like Haskell), higher-kinded types, and row polymorphism. Show the unification algorithm, constraint generation, and error reporting. Prove soundness of the inference.",
            "expected_tier": "deep",
        },
        {
            "query": "Analyze the theoretical limits of KV-cache compression for transformer models. Derive the information-theoretic lower bound on compression ratio given a target perplexity degradation. Then design an optimal compression scheme that approaches this bound. Compare against existing methods (quantization, pruning, clustering) and show where they fall short.",
            "expected_tier": "deep",
        },
        {
            "query": "Design a Byzantine fault-tolerant consensus protocol that achieves optimal message complexity O(n²) in the common case and O(n³) during view changes. Support dynamic membership changes. Prove safety and liveness. Compare with PBFT, HotStuff, and Tendermint. Implement the core protocol in pseudocode with detailed state machine.",
            "expected_tier": "deep",
        },
    ],
}


def generate(output_dir: str = "."):
    """Generate all benchmark dataset files."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    for tier, queries in DATASETS.items():
        filepath = out / f"{tier}.jsonl"
        with open(filepath, "w") as f:
            for i, q in enumerate(queries):
                record = {
                    "id": f"{tier}_{i:03d}",
                    "query": q["query"],
                    "expected_tier": q["expected_tier"],
                    "messages": [
                        {"role": "user", "content": q["query"]},
                    ],
                }
                f.write(json.dumps(record) + "\n")

        print(f"Generated {filepath} ({len(queries)} queries)")

    print(f"\nTotal: {sum(len(q) for q in DATASETS.values())} queries across {len(DATASETS)} tiers")


if __name__ == "__main__":
    generate()
