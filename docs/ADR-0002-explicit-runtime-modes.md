# ADR-0002: Explicit Runtime Modes

## Status

Accepted

## Decision

Runtime behavior is controlled by `IP_SAKTI_ENV` and `IP_SAKTI_TEST_MODE`.
Production mode must fail clearly when a configured model or external service is
unavailable. Test mode may use deterministic local providers and in-memory
stores, but health output must identify that mode.

## Rationale

The previous implementation silently replaced configured production services
with mock embeddings, in-memory indexes, and template generation. Silent
fallback makes a successful HTTP response misleading for legal research.

## Consequences

- Test substitutes are explicit and cannot become production defaults.
- Local smoke tests remain possible without paid credentials.
- Production startup validation becomes stricter.
