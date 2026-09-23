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

## Dry-running a chain against sample prompts

`chainroute simulate` runs a chain against a batch of prompts and reports what it would have
decided for each one — no proxy, no LLM calls, no cost:

```bash
echo '"this is urgent, please help"' > prompts.jsonl
echo '"my ssn is 123-45-6789"' >> prompts.jsonl
chainroute simulate --config config.yaml --group chat --chain chain.yaml --prompts prompts.jsonl
```

```
[1] 'this is urgent, please help'
  pin claude-sonnet-4 (keyword match: prefers claude-sonnet-4)
    openai/gpt-4o-mini        eligible
    anthropic/claude-sonnet-4 PINNED

[2] 'my ssn is 123-45-6789'
  narrowed to 1/2
    openai/gpt-4o-mini        EXCLUDED - looks like sensitive data; only ['azure'] are trusted
    anthropic/claude-sonnet-4 eligible
```

Each line in the prompts file is either a plain JSON string (shorthand for a one-message
conversation) or a full object — `{"messages": [...], "session_id": "...", "metadata": {...}}` —
for testing session-aware plugins like `StickySession`. Add `--json` for one JSON result object
per line instead, to feed into a script. If a plugin in the chain calls a real API (see
`ModerationGuard`), simulate calls it for real too — that's the point, seeing real results
without spending anything on the LLM call itself.

## Shadow mode

Before a new plugin or chain can affect real traffic, you can run it in shadow mode: every
plugin still runs and its verdict is still computed and logged, but the *original* deployment
list is what actually goes back to LiteLLM — nothing is enforced, and a would-be `Veto` is
logged instead of raised.

```yaml
# chain.yaml
mode: shadow
plugins:
  - use: plugins.MyNewPlugin
```

Flip it at deploy time without touching the file, with `CHAINROUTE_MODE=shadow` (or `=enforce`)
as an environment variable — this takes precedence over `chain.yaml`'s `mode:`, so you can turn
enforcement on or off the same way you'd flip any other feature flag. Watch for lines starting
`chainroute[shadow]:` in the proxy's logs to see what each plugin *would* have done.

## Timeouts

Every plugin hook runs under a timeout (2 seconds by default) — a plugin that hangs is treated
exactly like one that raises: logged, skipped for that request, never left to block a response
indefinitely. This matters most for a plugin that calls out to a real API (a moderation
endpoint, say); override the default per chain, or per plugin:

```python
ChainRoute(plugins=[...], default_timeout_s=5.0)   # SDK: the whole chain's default
```

```yaml
# chain.yaml: override for one plugin only
plugins:
  - use: plugins.CallsAModerationAPI
    timeout_s: 0.8
```

Or set `CHAINROUTE_DEFAULT_TIMEOUT_S` as an environment variable to change the chain-wide default
without touching `chain.yaml`.

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
| `ModerationGuard` | Checks the latest message against OpenAI's moderation endpoint; restricts to trusted providers (or rejects) when flagged. Fails open by default if the API errors or times out — see below. |
| `BanditRouter` | An epsilon-greedy multi-armed bandit across every deployment in the group: tries each at least once, then adaptively shifts traffic toward whichever is empirically winning (by success rate or by cost), while still exploring occasionally. |

Each is under 60 lines — read one as a template before writing your own.

`BanditRouter` is `WeightedCanary`'s fixed split turned adaptive: instead of a percentage you
set once, it learns from real `on_success`/`on_failure` outcomes which deployment is actually
performing best, and routes more traffic there over time. It optimizes across a whole group's
worth of *aggregate* traffic, not any one conversation, so it doesn't try to keep a session
consistent the way `WeightedCanary` does — add `StickySession` to the same chain (either order)
if you also want a session pinned to whatever the bandit picked for its first turn:

```yaml
plugins:
  - use: chainroute.plugins.bandit.BanditRouter
    with: {optimize_for: cost}
  - use: chainroute.plugins.sticky_session.StickySession
```

That composes correctly (the pin keeps winning even as the bandit's own preference drifts)
*because* `BanditRouter` expresses its choice as a `bias`, not an `exclude` — a pin always wins
over a bias at merge time, regardless of which plugin ran first. If you're writing a plugin
meant to express a mere *preference* rather than a hard rule, prefer `bias` over `exclude` for
the same reason: it's what lets it compose with something else's pin instead of silently
overriding it.

`ModerationGuard` is the first plugin that calls a real external API rather than a local check,
and needed no extra reliability code of its own to do it safely — it just relies on the timeout
and isolation every plugin already gets (see [Timeouts](#timeouts) above). The one thing worth
choosing deliberately for a *compliance* plugin specifically: `fail_closed: true` vetoes a
request rather than letting it through unmoderated if the API call itself fails, since chainroute's
usual fail-open default isn't always the right call for this one.

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

**Isolation**: a plugin that raises — or runs past its timeout, see below — is logged and
skipped for that request; one broken or slow plugin degrades to "did nothing," never to "took
the proxy down" or "added unbounded latency." `Veto` is the one exception that's allowed
through in enforce mode, since raising it is how a plugin says the request must stop.

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
