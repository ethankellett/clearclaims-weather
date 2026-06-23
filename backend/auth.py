# =============================================================================
#  Clerk authentication (optional per-user logins)
#
#  When CLERK_ISSUER is set, the /generate endpoint requires a valid Clerk
#  session token (sent by the signed-in browser). We verify the token's
#  signature against Clerk's public keys (JWKS) — no secret key needed on the
#  server, and it can't be forged.
#
#  Environment variables:
#    CLERK_ISSUER    your Clerk "Frontend API" / issuer URL, e.g.
#                    https://your-app.clerk.accounts.dev   (or your custom domain)
#    CLERK_JWKS_URL  optional override (defaults to <issuer>/.well-known/jwks.json)
# =============================================================================

import os

CLERK_ISSUER = os.environ.get("CLERK_ISSUER", "").strip().rstrip("/")
CLERK_JWKS_URL = os.environ.get("CLERK_JWKS_URL", "").strip()

_jwks_client = None


def clerk_enabled() -> bool:
    return bool(CLERK_ISSUER)


def _client():
    global _jwks_client
    if _jwks_client is None:
        from jwt import PyJWKClient
        url = CLERK_JWKS_URL or (CLERK_ISSUER + "/.well-known/jwks.json")
        _jwks_client = PyJWKClient(url)
    return _jwks_client


def verify_clerk(token: str | None):
    """Return the token's claims dict if valid, else None.

    Verifies signature (RS256) against Clerk's JWKS, checks expiry, and that the
    issuer matches CLERK_ISSUER. Clerk session tokens don't carry an audience by
    default, so audience verification is disabled.
    """
    if not token:
        return None
    try:
        import jwt
        signing_key = _client().get_signing_key_from_jwt(token)
        claims = jwt.decode(
            token,
            signing_key.key,
            algorithms=["RS256"],
            issuer=CLERK_ISSUER or None,
            options={"verify_aud": False},
        )
        return claims
    except Exception:
        return None


def user_label(claims: dict | None) -> str:
    """A human-ish identifier for audit/history (email if present, else user id)."""
    if not claims:
        return ""
    return (claims.get("email")
            or claims.get("email_address")
            or claims.get("sub")
            or "")
