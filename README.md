# ChainRoute

**A plugin chain for LiteLLM's routing decisions.**

[LiteLLM](https://github.com/BerriAI/litellm) already knows how to pick a deployment by cost,
latency, load, or a fallback chain. ChainRoute lets you add your own rules on top — small,
composable plugins that can prefer, exclude, pin, or reject a deployment for a specific
request — without forking LiteLLM or hand-rolling a custom `CustomLogger` every time you need
one more routing rule. It's the same idea as an io-filter chain, or the plugin-chain pattern
vLLM's and SGLang's own routers use internally, but as a small, independent package you drop
into an existing LiteLLM setup with one line.

This is a routing *decision* layer, not an observability dashboard — pair it with
[RouteLens](https://github.com/aaravbaphna/routelens) if you also want a dashboard showing why
each decision was made.

## Install

```bash
pip install chainroute
chainroute init                                          # scaffolds chain.yaml + plugins.py
chainroute install --config config.yaml --patch           # wires ChainRoute into your proxy config
```

Edit `chain.yaml` to build your chain, then restart the proxy.

```yaml
# chain.yaml
plugins:
  - use: chainroute.plugins.sticky_session.StickySession
  - use: chainroute.plugins.keyword.KeywordRoute
    with:
      rules:
        - match: "urgent|production is down|sev1"
          prefer_model: gpt-4o
```

`chainroute check --config config.yaml` validates a chain against your LiteLLM config before you
restart anything — it loads every plugin and cross-checks anything KeywordRoute references
against the deployments you've actually configured.

## Built-in plugins

Run `chainroute list` for the live list with descriptions. As of this writing:

| Plugin | What it does |
|---|---|
| `StickySession` | Keeps every turn of one conversation on the deployment that handled its first turn, instead of the routing strategy picking a new one each turn. |
| `KeywordRoute` | Prefers a specific deployment within the group when the message matches a regex (e.g. route "urgent" requests to a stronger model already in the same group). |
| `SensitiveDataGuard` | Restricts (or rejects) a request to a trusted set of providers when the message looks like it contains sensitive data (SSNs, card numbers, ...). |
| `TagAffinity` | Keeps only deployments tagged for a request's `metadata.tags`. |
| `WeightedCanary` | Sends a fixed, deterministic percentage of traffic to a candidate model — the same session always lands on the same side of the split. |
| `BudgetGuard` | Steers a caller toward a cheap allow-list once their spend crosses a cap in a rolling window; fails open if nothing cheap is eligible. |

Each is under 60 lines — read one as a template before writing your own.

## Writing a plugin

A plugin is a small class. Every hook is optional; implement only what you need:

```python
from chainroute import RoutingPlugin, RouteContext

class PreferCheaperAfterHours(RoutingPlugin):
    def configure(self, cheap_model="gpt-4o-mini", start_hour=9, end_hour=18):
        self.cheap_model, self.start_hour, self.end_hour = cheap_model, start_hour, end_hour

    async def apply(self, ctx: RouteContext, candidates):
        import datetime
        hour = datetime.datetime.now().hour
        if self.start_hour <= hour < self.end_hour:
            return  # business hours: no preference
        for c in candidates:
            if c.model == self.cheap_model:
                ctx.pin(c, "outside business hours")
                return
```

Save it as `plugins.py` next to your `chain.yaml` — no packaging, no `PYTHONPATH`, just
`use: plugins.PreferCheaperAfterHours` in the chain. `chainroute init` scaffolds exactly this.

**What a plugin can do**, on the `candidates` it's handed (one `Candidate` per deployment
already eligible in the request's model group):

- `candidate.exclude("reason")` — remove it from consideration.
- `candidate.bias(amount, "reason")` — nudge its score; only the top-scoring survivors continue
  to LiteLLM's own `routing_strategy`.
- `ctx.pin(candidate, "reason")` — force the whole request onto this one, skipping the rest of
  the chain's scoring and the configured strategy.
- `raise Veto("reason")` — reject the request outright (surfaces as `litellm.BadRequestError`).

A plugin that does none of these for a given request is a no-op for it, and ChainRoute falls
through to whatever `routing_strategy` you already have configured. An empty chain is a
byte-for-byte no-op.

There's also `on_success(ctx, chosen, cost)` / `on_failure(ctx, attempted, error_class)`, which
fire after the fact — the only place a plugin reliably learns which deployment was actually used
(`StickySession` and `BudgetGuard` both need this).

**Scope**: a plugin arbitrates *within* one model group (everything under one `model_name` in
your `model_list`) — it can't send a request to a different group. LiteLLM has no public hook
for that outside its own built-in semantic/complexity auto-routers; put the deployments you want
to choose between under one `model_name` and pin between them (see `KeywordRoute`).

**Isolation**: a plugin that raises is logged and skipped for that request — one broken plugin
degrades to "did nothing," never to "took the proxy down." `Veto` is the one exception that's
allowed through, since raising it is how a plugin says the request must stop.

## Using the SDK `Router` directly (no proxy)

```python
import litellm
from litellm import Router
from chainroute import ChainRoute
from chainroute.plugins.sticky_session import StickySession

litellm.callbacks = [ChainRoute(plugins=[StickySession()])]  # or plugins=None to load a chain.yaml
router = Router(model_list=[...])
```

## How it works

ChainRoute is a LiteLLM `CustomLogger`. It implements:

- `async_filter_deployments` — runs right before the routing strategy picks a winner; this is
  where a chain's plugins see the eligible candidates and exclude, bias, or pin one.
- `async_log_success_event` / `async_log_failure_event` — fire each plugin's `on_success` /
  `on_failure` with the deployment that was actually used.
- `async_pre_call_hook` (proxy only) — stamps a per-request id, so retries and fallbacks within
  one HTTP request are told apart from the next request that happens to share a trace id.

## Development

```bash
pip install -e '.[dev]'
pytest
```

`tests/test_callback_sdk.py` runs the plugins against a real `litellm.Router` (via
`mock_response`, no network or API keys) to prove they actually change routing outcomes, not
just that the logic looks right in isolation.

## License

MIT
