"""
Authentication and Authorization Module
Phase 18: JWT/OIDC authentication, API keys, rate limiting.
"""

from __future__ import annotations
import hashlib
import secrets
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set
from uuid import uuid4

from fastapi import Depends, Header, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer, OAuth2PasswordBearer
from jose import JWTError, jwt
from pydantic import BaseModel

from ip_sakti.config.loader import Settings, get_settings
from ip_sakti.api.models import (
    APIKeyCreate,
    APIKeyListResponse,
    APIKeyResponse,
    ErrorDetail,
    ErrorResponse,
    TokenRequest,
    TokenResponse,
)


# ============================================================
# JWT AUTHENTICATION
# ============================================================

class JWTManager:
    """JWT token management."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.algorithm = settings.jwt_algorithm
        self.issuer = settings.jwt_issuer
        self.audience = settings.jwt_audience
        self.access_token_ttl = settings.jwt_access_token_ttl
        self.refresh_token_ttl = settings.jwt_refresh_token_ttl
        self._private_key: Optional[str] = None
        self._public_key: Optional[str] = None
        self._load_keys()

    def _load_keys(self) -> None:
        """Load RSA keys for RS256."""
        # In production, load from secure key management
        # For development, generate or use fixed keys
        import os
        private_key_path = os.getenv("JWT_PRIVATE_KEY_PATH")
        public_key_path = os.getenv("JWT_PUBLIC_KEY_PATH")

        if private_key_path and os.path.exists(private_key_path):
            with open(private_key_path, "r") as f:
                self._private_key = f.read()
        if public_key_path and os.path.exists(public_key_path):
            with open(public_key_path, "r") as f:
                self._public_key = f.read()

        # Fallback for development
        if not self._private_key or not self._public_key:
            # Use a shared secret for HS256 in development
            if self.algorithm == "RS256":
                # Generate a dev key pair (in production, use proper key management)
                from cryptography.hazmat.primitives.asymmetric import rsa
                from cryptography.hazmat.primitives import serialization
                private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
                self._private_key = private_key.private_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PrivateFormat.PKCS8,
                    encryption_algorithm=serialization.NoEncryption()
                ).decode()
                self._public_key = private_key.public_key().public_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PublicFormat.SubjectPublicKeyInfo
                ).decode()

    def create_access_token(
        self,
        subject: str,
        scopes: List[str],
        user_id: str,
        roles: List[str] = None,
        additional_claims: Dict[str, Any] = None,
    ) -> str:
        """Create JWT access token."""
        now = datetime.utcnow()
        expire = now + timedelta(seconds=self.access_token_ttl)

        claims = {
            "sub": subject,
            "user_id": user_id,
            "scopes": scopes,
            "roles": roles or [],
            "iat": now,
            "exp": expire,
            "iss": self.issuer,
            "aud": self.audience,
            "jti": str(uuid4()),
        }

        if additional_claims:
            claims.update(additional_claims)

        return jwt.encode(claims, self._private_key, algorithm=self.algorithm)

    def create_refresh_token(self, subject: str, user_id: str) -> str:
        """Create JWT refresh token."""
        now = datetime.utcnow()
        expire = now + timedelta(seconds=self.refresh_token_ttl)

        claims = {
            "sub": subject,
            "user_id": user_id,
            "type": "refresh",
            "iat": now,
            "exp": expire,
            "iss": self.issuer,
            "aud": self.audience,
            "jti": str(uuid4()),
        }

        return jwt.encode(claims, self._private_key, algorithm=self.algorithm)

    def verify_token(self, token: str) -> Dict[str, Any]:
        """Verify and decode JWT token."""
        try:
            payload = jwt.decode(
                token,
                self._public_key,
                algorithms=[self.algorithm],
                audience=self.audience,
                issuer=self.issuer,
            )
            return payload
        except JWTError as e:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"Invalid token: {str(e)}",
                headers={"WWW-Authenticate": "Bearer"},
            )

    def decode_token_unverified(self, token: str) -> Dict[str, Any]:
        """Decode token without verification (for debugging)."""
        return jwt.decode(token, options={"verify_signature": False})


# ============================================================
# API KEY MANAGEMENT
# ============================================================

class APIKeyManager:
    """API key management with hashing and rate limiting tiers."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.hash_algorithm = settings.api_key_hash_algorithm
        self.default_ttl_days = settings.api_key_default_ttl_days
        self.max_keys_per_user = settings.api_key_max_keys_per_user
        self.rate_limit_tiers = settings.rate_limit_tiers
        # In production, use Redis or database
        self._keys: Dict[str, Dict[str, Any]] = {}  # key_hash -> key_data
        self._user_keys: Dict[str, Set[str]] = {}   # user_id -> set of key_hashes

    def _hash_key(self, key: str) -> str:
        """Hash API key for storage."""
        if self.hash_algorithm == "sha256":
            return hashlib.sha256(key.encode()).hexdigest()
        # Add other algorithms as needed
        return hashlib.sha256(key.encode()).hexdigest()

    def _generate_key(self) -> str:
        """Generate a new API key."""
        # Format: ipsk_[32 random chars]
        return f"ipsk_{secrets.token_urlsafe(32)}"

    def create_key(
        self,
        user_id: str,
        name: str,
        scopes: List[str],
        expires_in_days: Optional[int] = None,
        rate_limit_tier: str = "standard",
    ) -> APIKeyResponse:
        """Create a new API key."""
        # Check user key limit
        user_key_count = len(self._user_keys.get(user_id, set()))
        if user_key_count >= self.max_keys_per_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Maximum API keys per user ({self.max_keys_per_user}) reached",
            )

        # Validate rate limit tier
        if rate_limit_tier not in self.rate_limit_tiers:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid rate limit tier: {rate_limit_tier}",
            )

        # Generate key
        raw_key = self._generate_key()
        key_hash = self._hash_key(raw_key)
        key_id = str(uuid4())[:8]
        now = datetime.utcnow()
        expires_at = now + timedelta(days=expires_in_days or self.default_ttl_days)

        key_data = {
            "key_id": key_id,
            "key_hash": key_hash,
            "user_id": user_id,
            "name": name,
            "scopes": scopes,
            "rate_limit_tier": rate_limit_tier,
            "created_at": now,
            "expires_at": expires_at,
            "last_used": None,
            "is_active": True,
        }

        self._keys[key_hash] = key_data
        if user_id not in self._user_keys:
            self._user_keys[user_id] = set()
        self._user_keys[user_id].add(key_hash)

        return APIKeyResponse(
            key_id=key_id,
            name=name,
            key=raw_key,  # Only returned once!
            scopes=scopes,
            rate_limit_tier=rate_limit_tier,
            created_at=now,
            expires_at=expires_at,
            last_used=None,
        )

    def verify_key(self, key: str) -> Optional[Dict[str, Any]]:
        """Verify API key and return key data."""
        key_hash = self._hash_key(key)
        key_data = self._keys.get(key_hash)

        if not key_data:
            return None

        if not key_data["is_active"]:
            return None

        if key_data["expires_at"] and datetime.utcnow() > key_data["expires_at"]:
            return None

        # Update last used
        key_data["last_used"] = datetime.utcnow()
        return key_data

    def list_keys(self, user_id: str) -> APIKeyListResponse:
        """List all API keys for a user (without raw keys)."""
        key_hashes = self._user_keys.get(user_id, set())
        keys = []
        for key_hash in key_hashes:
            key_data = self._keys.get(key_hash)
            if key_data:
                keys.append(APIKeyResponse(
                    key_id=key_data["key_id"],
                    name=key_data["name"],
                    key="***",  # Never return raw key again
                    scopes=key_data["scopes"],
                    rate_limit_tier=key_data["rate_limit_tier"],
                    created_at=key_data["created_at"],
                    expires_at=key_data["expires_at"],
                    last_used=key_data["last_used"],
                ))
        return APIKeyListResponse(keys=keys, total=len(keys))

    def revoke_key(self, user_id: str, key_id: str) -> bool:
        """Revoke an API key."""
        key_hashes = self._user_keys.get(user_id, set())
        for key_hash in key_hashes:
            key_data = self._keys.get(key_hash)
            if key_data and key_data["key_id"] == key_id:
                key_data["is_active"] = False
                return True
        return False

    def get_rate_limit(self, tier: str) -> Dict[str, int]:
        """Get rate limit configuration for tier."""
        return self.rate_limit_tiers.get(tier, self.rate_limit_tiers.get("standard", {}))


# ============================================================
# RATE LIMITING
# ============================================================

class RateLimiter:
    """Token bucket rate limiter."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.api_key_manager = APIKeyManager(settings)
        # In production, use Redis for distributed rate limiting
        self._buckets: Dict[str, Dict[str, Any]] = {}  # key -> {tokens, last_refill}

    def _get_bucket_key(self, identifier: str, endpoint: str = "") -> str:
        """Generate bucket key."""
        return f"ratelimit:{identifier}:{endpoint}"

    def _refill_bucket(self, bucket: Dict[str, Any], capacity: int, refill_rate: float) -> None:
        """Refill token bucket."""
        now = time.time()
        elapsed = now - bucket["last_refill"]
        bucket["tokens"] = min(capacity, bucket["tokens"] + elapsed * refill_rate)
        bucket["last_refill"] = now

    def check_rate_limit(
        self,
        identifier: str,
        endpoint: str = "",
        tier: str = "standard",
        cost: int = 1,
    ) -> tuple[bool, Dict[str, Any]]:
        """Check and consume rate limit."""
        tier_config = self.api_key_manager.get_rate_limit(tier)
        capacity = tier_config.get("requests_per_minute", 60)
        refill_rate = capacity / 60.0  # tokens per second

        bucket_key = self._get_bucket_key(identifier, endpoint)
        bucket = self._buckets.get(bucket_key, {"tokens": capacity, "last_refill": time.time()})

        self._refill_bucket(bucket, capacity, refill_rate)

        if bucket["tokens"] >= cost:
            bucket["tokens"] -= cost
            self._buckets[bucket_key] = bucket
            return True, {
                "limit": capacity,
                "remaining": int(bucket["tokens"]),
                "reset": int(bucket["last_refill"] + (capacity - bucket["tokens"]) / refill_rate),
            }

        self._buckets[bucket_key] = bucket
        return False, {
            "limit": capacity,
            "remaining": 0,
            "reset": int(bucket["last_refill"] + (capacity - bucket["tokens"]) / refill_rate),
        }


# ============================================================
# DEPENDENCIES
# ============================================================

# Security schemes
bearer_scheme = HTTPBearer(auto_error=False)
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token", auto_error=False)


# Global instances (initialized in app startup)
_jwt_manager: Optional[JWTManager] = None
_api_key_manager: Optional[APIKeyManager] = None
_rate_limiter: Optional[RateLimiter] = None


def get_jwt_manager() -> JWTManager:
    global _jwt_manager
    if _jwt_manager is None:
        _jwt_manager = JWTManager(get_settings())
    return _jwt_manager


def get_api_key_manager() -> APIKeyManager:
    global _api_key_manager
    if _api_key_manager is None:
        _api_key_manager = APIKeyManager(get_settings())
    return _api_key_manager


def get_rate_limiter() -> RateLimiter:
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = RateLimiter(get_settings())
    return _rate_limiter


# Helper to set global instances from app lifespan
def set_global_jwt_manager(manager: JWTManager) -> None:
    global _jwt_manager
    _jwt_manager = manager


def set_global_api_key_manager(manager: APIKeyManager) -> None:
    global _api_key_manager
    _api_key_manager = manager


def set_global_rate_limiter(limiter: RateLimiter) -> None:
    global _rate_limiter
    _rate_limiter = limiter


# ============================================================
# AUTH DEPENDENCIES
# ============================================================

class CurrentUser(BaseModel):
    """Current authenticated user."""
    user_id: str
    subject: str
    scopes: List[str]
    roles: List[str]
    auth_method: str  # "jwt", "api_key"


async def get_current_user_jwt(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
    jwt_manager: JWTManager = Depends(get_jwt_manager),
) -> Optional[CurrentUser]:
    """Get current user from JWT token."""
    if not credentials:
        return None

    try:
        payload = jwt_manager.verify_token(credentials.credentials)
    except HTTPException:
        return None

    return CurrentUser(
        user_id=payload.get("user_id", ""),
        subject=payload.get("sub", ""),
        scopes=payload.get("scopes", []),
        roles=payload.get("roles", []),
        auth_method="jwt",
    )


async def get_current_user_api_key(
    request: Request,
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    api_key_manager: APIKeyManager = Depends(get_api_key_manager),
) -> Optional[CurrentUser]:
    """Get current user from API key."""
    # Check header first, then query parameter
    api_key = x_api_key or request.query_params.get("api_key")

    if not api_key:
        return None

    key_data = api_key_manager.verify_key(api_key)
    if not key_data:
        return None

    return CurrentUser(
        user_id=key_data["user_id"],
        subject=key_data["user_id"],
        scopes=key_data["scopes"],
        roles=[],
        auth_method="api_key",
    )


async def get_current_user(
    request: Request,
    jwt_user: Optional[CurrentUser] = Depends(get_current_user_jwt),
    api_key_user: Optional[CurrentUser] = Depends(get_current_user_api_key),
) -> CurrentUser:
    """Get current user from either JWT or API key."""
    # JWT takes precedence if both provided
    if jwt_user:
        return jwt_user
    if api_key_user:
        return api_key_user

    # Local UI smoke tests may opt into a clearly named bypass. Never enable
    # this implicitly in production.
    import os
    if os.getenv("IP_SAKTI_TEST_MODE", "false").lower() in {"1", "true", "yes", "on"}:
        return CurrentUser(
            user_id="test-user",
            subject="test-user",
            scopes=["query", "query:read", "ingest", "ingest:write", "admin"],
            roles=["admin"],
            auth_method="test_mode",
        )

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Authentication required",
        headers={"WWW-Authenticate": "Bearer, API-Key"},
    )


def require_scopes(*required_scopes: str):
    """Dependency factory to require specific scopes."""
    async def check_scopes(user: CurrentUser = Depends(get_current_user)) -> CurrentUser:
        user_scopes = set(user.scopes)
        if not all(scope in user_scopes for scope in required_scopes):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Required scopes: {', '.join(required_scopes)}",
            )
        return user
    return check_scopes


def require_roles(*required_roles: str):
    """Dependency factory to require specific roles."""
    async def check_roles(user: CurrentUser = Depends(get_current_user)) -> CurrentUser:
        user_roles = set(user.roles)
        if not any(role in user_roles for role in required_roles):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Required roles: {', '.join(required_roles)}",
            )
        return user
    return check_roles


# ============================================================
# RATE LIMIT DEPENDENCY
# ============================================================

async def rate_limit_dependency(
    request: Request,
    user: CurrentUser = Depends(get_current_user),
    rate_limiter: RateLimiter = Depends(get_rate_limiter),
) -> None:
    """Apply rate limiting based on user's tier."""
    # Get tier from user (would be stored in user profile in production)
    tier = "standard"  # Default

    # For API key users, get tier from key data
    if user.auth_method == "api_key":
        # Would need to look up the key data
        pass

    allowed, limit_info = rate_limiter.check_rate_limit(
        identifier=user.user_id,
        endpoint=request.url.path,
        tier=tier,
    )

    # Add rate limit headers
    request.state.rate_limit = limit_info

    if not allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded",
            headers={
                "X-RateLimit-Limit": str(limit_info["limit"]),
                "X-RateLimit-Remaining": str(limit_info["remaining"]),
                "X-RateLimit-Reset": str(limit_info["reset"]),
            },
        )


# ============================================================
# MIDDLEWARE
# ============================================================

async def rate_limit_middleware(request: Request, call_next):
    """Middleware to add rate limit headers to all responses."""
    response = await call_next(request)

    if hasattr(request.state, "rate_limit"):
        limit_info = request.state.rate_limit
        response.headers["X-RateLimit-Limit"] = str(limit_info["limit"])
        response.headers["X-RateLimit-Remaining"] = str(limit_info["remaining"])
        response.headers["X-RateLimit-Reset"] = str(limit_info["reset"])

    return response
