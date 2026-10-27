"""Offline documentation smoke test. Python is optional; the engine is Rust.

No HTTP client, credentials, platform calls, or real financial measurements.
Use a NEW output directory. Logs/checkpoints are synthetic demonstration output,
not a production durable simulation ledger.
"""
import argparse
import hashlib
import json
from pathlib import Path
import queue
import subprocess
import threading


FIXTURES = Path(__file__).resolve().parent


def load(name):
    return json.loads((FIXTURES / name).read_text())


def toy_score(expression):
    return -min(1 + expression.count("("), 20) / 20


class Bridge:
    def __init__(self, binary):
        self.process = subprocess.Popen(
            [str(binary)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            text=True, encoding="utf-8",
        )
        self.lines = queue.Queue()
        self.requests, self.responses = [], []
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.reader.start()

    def _read(self):
        for line in self.process.stdout:
            self.lines.put(line)
        self.lines.put(None)

    def rpc(self, **request):
        self.requests.append(request)
        self.process.stdin.write(json.dumps(request) + "\n")
        self.process.stdin.flush()
        line = self.lines.get(timeout=15)
        if line is None:
            raise RuntimeError("JSONL bridge exited before its response")
        response = json.loads(line)
        self.responses.append(response)
        if "error" in response:
            raise RuntimeError(response["error"])
        return response

    def close(self):
        self.process.stdin.close()
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=5)
        self.reader.join(timeout=5)
        self.process.stdout.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cli", type=Path, required=True)
    parser.add_argument("--bridge", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.cli, args.bridge = args.cli.resolve(), args.bridge.resolve()
    if not args.cli.is_file() or not args.bridge.is_file():
        parser.error("build the CLI and measured_search_stdio example first")
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)

    def save(name, value):
        (out / name).write_text(json.dumps(value, indent=2) + "\n")

    def cli(*arguments, success=True):
        result = subprocess.run([str(args.cli), *map(str, arguments)],
                                text=True, capture_output=True, timeout=90)
        if (result.returncode == 0) != success:
            raise RuntimeError(result.stderr or result.stdout)
        return json.loads(result.stdout) if success else result.stderr

    catalog, config, seeds = load("catalog.json"), load("session-config.json"), load("seeds.json")
    context = load("evaluation-context.json")
    assert context["objective_version"] == config["evaluator_context"]
    assert all(seed["outcome"] == toy_score(seed["expression"]) for seed in seeds)
    candidates = out / "candidates.jsonl"
    save("search.json", cli("search", "--catalog", FIXTURES / "catalog.json",
         "--output", candidates, "--max-candidates", 256, "--duration-seconds", 60,
         "--threads", 2, "--seed", 7))
    save("compiled.json", cli("compile", "rank(negate(returns))",
         "--dialect", FIXTURES / "dialect.json"))
    assert "does not allow field" in cli("compile", "rank(high)",
         "--dialect", FIXTURES / "dialect.json", success=False)
    save("holdout.json", cli("audit-feedback", "--feedback", FIXTURES / "feedback-training.json",
         "--evaluation-feedback", FIXTURES / "feedback-evaluation.json"))
    assert "overlap" in cli("audit-feedback", "--feedback", FIXTURES / "feedback-training.json",
         "--evaluation-feedback", FIXTURES / "feedback-training.json", success=False)
    cli("screen-feedback", "--input", candidates,
        "--feedback", FIXTURES / "feedback-training.json", "--output", out / "screening.json",
        "--minimum-conservative-outcome", 0)

    bridge = Bridge(args.bridge)
    restored = None
    try:
        initialized = bridge.rpc(op="init_measured", config=config, catalog=catalog,
            evaluator_context=config["evaluator_context"], pool_context=context["pool_context"],
            seeds=seeds, excluded=[])
        assert initialized["imported"] == 2
        assert initialized["measured"] == 0
        batches = 0
        while True:
            pending = bridge.rpc(op="ask")
            assert pending == bridge.rpc(op="ask"), "pending retry changed identities"
            if not pending["proposals"]:
                break
            all_pending = {p["semantic_fingerprint"] for p in pending["proposals"]}
            chosen = {p["semantic_fingerprint"] for p in pending["proposals"][:2]}
            # Incumbents must also have outcomes under the same objective/context.
            requests = [p for p in pending["score_requests"]
                        if p["semantic_fingerprint"] not in all_pending or p["semantic_fingerprint"] in chosen]
            outcomes = [dict(semantic_fingerprint=p["semantic_fingerprint"],
                             outcome=toy_score(p["expression"])) for p in requests]
            result = bridge.rpc(op="tell_screened", evaluator_context=config["evaluator_context"],
                                pool_context=context["pool_context"], outcomes=outcomes)
            checkpoint = bridge.rpc(op="checkpoint")["checkpoint"]
            save("checkpoint.json", json.loads(checkpoint))
            restored = Bridge(args.bridge)
            restored.rpc(op="restore", config=config, catalog=catalog, checkpoint=checkpoint)
            assert restored.rpc(op="checkpoint")["checkpoint"] == checkpoint
            assert restored.rpc(op="ask") == bridge.rpc(op="ask")
            restored.close()
            restored = None
            batches += 1
        assert batches == 4 and result["measured"] == 8 and result["screened_out"] == 24
        for name, rows in (("requests.jsonl", bridge.requests), ("responses.jsonl", bridge.responses)):
            (out / name).write_text("".join(json.dumps(r) + "\n" for r in rows))
        summary = dict(schema=1, market_evidence=False, external_requests=0,
                       batches=batches, imported=result["imported"], measured=result["measured"],
                       screened_out=result["screened_out"], restore_parity=True,
                       fixture_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                       for p in sorted(FIXTURES.glob("*.json"))})
        save("summary.json", summary)
        print(json.dumps(summary))
    finally:
        if restored is not None:
            restored.close()
        bridge.close()


if __name__ == "__main__":
    main()
