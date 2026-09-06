"""
IP-SAKTI Sahayak - Source Authority System
Phase 5: Source Authority System Implementation (from phase11-source-authority-system)

Assigns, maintains, and propagates authority tiers to every legal source,
document, chunk, and claim in the system.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Optional, List, Dict, Any, Callable
from enum import Enum
import re

from ip_sakti.core.models import (
    AuthorityTier, SourceType, AuthorityProfile, ChunkAuthority, ClaimAuthority,
    LegalChunk, HierarchyPath, JurisdictionCode, CitationRef, CitationType
)


# ============================================================
# AUTHORITY CLASSIFIER
# ============================================================

class AuthorityClassifier:
    """
    Classifies source documents into authority tiers.
    Uses deterministic rules based on source metadata.
    """
    
    # Deterministic classification rules - order matters (first match wins)
    CLASSIFICATION_RULES: List[tuple[Callable[[Dict], bool], AuthorityTier]] = [
        # TIER 1: Primary Legislation
        (lambda m: m.get("source_type") in ["act", "amendment_act", "ordinance", "constitution", "treaty", "protocol", "cop_decision"]
                    and m.get("gazette_published", False), AuthorityTier.TIER_1),
        
        # TIER 2: Subordinate Legislation (gazetted)
        (lambda m: m.get("source_type") in ["rule", "regulation", "notification", "order"]
                    and m.get("gazette_published", False), AuthorityTier.TIER_2),
        
        # TIER 3: Binding Judicial
        (lambda m: m.get("source_type") == "supreme_court_judgment", AuthorityTier.TIER_3),
        (lambda m: m.get("source_type") == "high_court_judgment"
                    and m.get("is_reported", False), AuthorityTier.TIER_3),
        
        # TIER 4: Quasi-Judicial/Regulatory
        (lambda m: m.get("source_type") in ["tribunal_order", "regulatory_decision"], AuthorityTier.TIER_4),
        (lambda m: m.get("source_type") == "circular"
                    and m.get("issuing_authority") in ["CDSCO", "FSSAI", "NBA", "IP_INDIA", "DGFT", "RBI", "SEBI", "IRDAI"], AuthorityTier.TIER_4),
        
        # TIER 5: Official Guidance
        (lambda m: m.get("source_type") in ["guideline", "faq", "pharmacopoeia", "formulary", "manual"], AuthorityTier.TIER_5),
        (lambda m: m.get("source_type") == "circular"
                    and m.get("issuing_authority") not in ["CDSCO", "FSSAI", "NBA", "IP_INDIA", "DGFT", "RBI", "SEBI", "IRDAI"], AuthorityTier.TIER_5),
        
        # TIER 6: Secondary Sources
        (lambda m: m.get("source_type") in ["academic_article", "commentary", "blog", "news", "journal"], AuthorityTier.TIER_6),
        
        # Registry records - factual, high authority
        (lambda m: m.get("source_type") in ["patent_record", "trademark_record", "design_record", "gi_record", "registry_record"], AuthorityTier.TIER_3),
    ]
    
    # Precedence within tier (lower = higher precedence)
    PRECEDENCE_RULES: Dict[AuthorityTier, Dict[str, int]] = {
        AuthorityTier.TIER_1: {
            "constitution": 1,
            "treaty": 2,
            "protocol": 3,
            "cop_decision": 4,
            "act": 5,
            "amendment_act": 6,
            "ordinance": 7,
        },
        AuthorityTier.TIER_2: {
            "rule": 1,
            "regulation": 2,
            "notification": 3,
            "order": 4,
        },
        AuthorityTier.TIER_3: {
            "supreme_court_judgment": 1,
            "high_court_judgment": 2,
            "patent_record": 3,
            "trademark_record": 4,
            "design_record": 5,
            "gi_record": 6,
            "registry_record": 7,
        },
        AuthorityTier.TIER_4: {
            "tribunal_order": 1,
            "regulatory_decision": 2,
            "circular": 3,
        },
        AuthorityTier.TIER_5: {
            "pharmacopoeia": 1,
            "formulary": 2,
            "guideline": 3,
            "manual": 4,
            "faq": 5,
            "circular": 6,
        },
    }
    
    def classify(self, metadata: Dict[str, Any]) -> AuthorityProfile:
        """Classify a source document into authority tier."""
        # Apply classification rules in order
        for condition, tier in self.CLASSIFICATION_RULES:
            if condition(metadata):
                authority_tier = tier
                break
        else:
            authority_tier = AuthorityTier.TIER_6
        
        # Compute precedence rank
        source_type = metadata.get("source_type", "unknown")
        precedence = self.PRECEDENCE_RULES.get(authority_tier, {}).get(source_type, 999)
        
        # Parse dates
        pub_date = metadata.get("publication_date")
        if isinstance(pub_date, str):
            pub_date = date.fromisoformat(pub_date)
        
        gazette_date = metadata.get("gazette_date")
        if isinstance(gazette_date, str):
            gazette_date = date.fromisoformat(gazette_date)
        
        return AuthorityProfile(
            source_id=metadata["source_id"],
            source_type=SourceType(source_type),
            authority_tier=authority_tier,
            jurisdiction_id=metadata.get("jurisdiction_id", "INDIA"),
            authority_name=metadata.get("authority_name", "Unknown"),
            enacting_authority=metadata.get("enacting_authority", "Unknown"),
            publication_date=pub_date or date.today(),
            gazette_number=metadata.get("gazette_number"),
            gazette_date=gazette_date,
            is_official_gazette=metadata.get("gazette_published", False),
            is_binding=authority_tier.value <= 4,
            precedence_rank=precedence,
            parent_source_id=metadata.get("parent_source_id"),
            amendment_history=metadata.get("amendment_history", []),
            superseded_by=metadata.get("superseded_by"),
            metadata=metadata
        )


# ============================================================
# AUTHORITY PROPAGATOR
# ============================================================

class AuthorityPropagator:
    """
    Propagates authority from document to chunks with adjustments.
    """
    
    def __init__(self, classifier: AuthorityClassifier):
        self.classifier = classifier
    
    def propagate(self, source_profile: AuthorityProfile, chunks: List[LegalChunk]) -> List[ChunkAuthority]:
        """Assign authority to each chunk from its source document."""
        chunk_authorities = []
        
        for chunk in chunks:
            # Base authority from source
            base_tier = source_profile.authority_tier
            base_weight = self._tier_weight(base_tier)
            
            # Adjustments
            adjustments = []
            
            # 1. Recency adjustment (newer = slightly higher for same tier)
            recency_factor = self._recency_factor(source_profile.publication_date)
            adjustments.append(("recency", recency_factor))
            
            # 2. Completeness adjustment (full section vs fragment)
            completeness = self._completeness_factor(chunk)
            adjustments.append(("completeness", completeness))
            
            # 3. Amendment status
            if source_profile.superseded_by:
                adjustments.append(("superseded", 0.3))
            elif source_profile.amendment_history:
                adjustments.append(("amended", 0.9))
            
            # 4. Chunk type adjustment
            chunk_type_factor = self._chunk_type_factor(chunk.chunk_type)
            adjustments.append(("chunk_type", chunk_type_factor))
            
            # 5. Citation density (chunks citing many authorities may be more authoritative)
            citation_factor = self._citation_density_factor(chunk)
            adjustments.append(("citation_density", citation_factor))
            
            # Compute final weight
            final_weight = base_weight
            for _, factor in adjustments:
                final_weight *= factor
            
            # Determine if this chunk represents amended version
            is_amended = bool(source_profile.amendment_history)
            amendment_date = None
            if is_amended:
                amendment_dates = []
                for aid in source_profile.amendment_history:
                    # Would fetch amendment date from registry
                    pass
                amendment_date = max(amendment_dates) if amendment_dates else None
            
            chunk_authorities.append(ChunkAuthority(
                chunk_id=chunk.chunk_id,
                source_id=source_profile.source_id,
                authority_tier=base_tier,
                authority_weight=min(1.0, final_weight),
                inheritance_path=[source_profile.source_id] + 
                                ([source_profile.parent_source_id] if source_profile.parent_source_id else []),
                is_amended_version=is_amended,
                amendment_date=amendment_date,
                metadata={
                    "source_type": source_profile.source_type.value,
                    "adjustments": dict(adjustments),
                    "chunk_type": chunk.chunk_type.value,
                    "hierarchy_level": chunk.hierarchy.get_level()
                }
            ))
        
        return chunk_authorities
    
    def _tier_weight(self, tier: AuthorityTier) -> float:
        return {1: 1.0, 2: 0.9, 3: 0.95, 4: 0.7, 5: 0.5, 6: 0.2}[tier.value]
    
    def _recency_factor(self, pub_date: date) -> float:
        """Slight boost for recent sources (within 5 years)."""
        years_old = (date.today() - pub_date).days / 365
        if years_old <= 1:
            return 1.02
        elif years_old <= 3:
            return 1.01
        elif years_old <= 5:
            return 1.0
        else:
            return 0.99  # Slight decay for very old sources
    
    def _completeness_factor(self, chunk: LegalChunk) -> float:
        """Full sections score higher than fragments."""
        level = chunk.hierarchy.get_level()
        if level == 3 and chunk.hierarchy.section:  # Full section
            return 1.0
        elif level == 4:  # Subsection
            return 0.95
        elif level >= 5:  # Clause/subclause
            return 0.9
        elif chunk.chunk_type == ChunkType.DEFINITION:
            return 1.05  # Definitions are high value
        elif chunk.chunk_type == ChunkType.TABLE:
            return 0.85
        else:
            return 0.85
    
    def _chunk_type_factor(self, chunk_type: ChunkType) -> float:
        """Adjust weight by chunk type."""
        factors = {
            ChunkType.DEFINITION: 1.05,
            ChunkType.SECTION: 1.0,
            ChunkType.SUBSECTION: 0.98,
            ChunkType.CLAUSE: 0.95,
            ChunkType.SUBCLAUSE: 0.92,
            ChunkType.TABLE: 0.9,
            ChunkType.SCHEDULE: 0.9,
            ChunkType.FOOTNOTE: 0.7,
            ChunkType.PREAMBLE: 0.8,
            ChunkType.AMENDMENT_NOTE: 0.85,
            ChunkType.ACT_METADATA: 0.8,
            ChunkType.REGISTRY_RECORD: 1.0,
            ChunkType.CASE_HEADNOTE: 1.02,
            ChunkType.CASE_PARAGRAPH: 0.95,
            ChunkType.TREATY_ARTICLE: 1.0,
            ChunkType.ANNEX: 0.85,
        }
        return factors.get(chunk_type, 0.9)
    
    def _citation_density_factor(self, chunk: LegalChunk) -> float:
        """Chunks with many internal citations may be interpretive provisions."""
        if not chunk.citations:
            return 1.0
        # Moderate boost for chunks that cite other provisions (interpretive)
        density = len(chunk.citations) / max(chunk.token_count / 100, 1)
        if density > 0.5:
            return 1.02
        return 1.0


# ============================================================
# AUTHORITY-WEIGHTED RETRIEVAL
# ============================================================

class AuthorityWeightedRetrieval:
    """
    Integrates authority weights into retrieval scoring.
    """
    
    def __init__(self, authority_store: AuthorityStore):
        self.authority_store = authority_store
    
    def compute_authority_score(self, chunk_id: str, query_jurisdiction: str) -> float:
        """Compute authority score for a chunk given query jurisdiction."""
        chunk_auth = self.authority_store.get_chunk_authority(chunk_id)
        if not chunk_auth:
            return 0.5  # Default
        
        # Base authority weight
        score = chunk_auth.authority_weight
        
        # Jurisdiction match bonus
        if chunk_auth.inheritance_path:
            # Check if any authority in path matches query jurisdiction
            # This requires mapping authority_name to jurisdiction
            pass  # Handled by Jurisdiction Engine hard filter
        
        # Binding vs non-binding
        if chunk_auth.authority_tier.value <= 4:
            score *= 1.1  # Binding authority boost
        
        return min(1.0, score)
    
    def rerank_by_authority(self, results: List[RetrievedChunk], query_jurisdiction: str) -> List[RetrievedChunk]:
        """Re-rank retrieval results by authority-weighted score."""
        for result in results:
            auth_score = self.compute_authority_score(result.chunk.chunk_id, query_jurisdiction)
            # Combine with retrieval score (authority gets 30% weight)
            result.combined_score = 0.7 * result.retrieval_score + 0.3 * auth_score
        
        results.sort(key=lambda r: r.combined_score, reverse=True)
        # Update ranks
        for i, result in enumerate(results):
            result.rank = i + 1
        return results


def authority_aware_rrf(bm25_results: List[str], dense_results: List[str], 
                       authority_scores: Dict[str, float], k: int = 60) -> List[str]:
    """
    RRF with authority weighting.
    Authority acts as a prior that modulates reciprocal rank.
    """
    from collections import defaultdict
    doc_scores = defaultdict(float)
    
    # BM25 contribution
    for rank, doc_id in enumerate(bm25_results):
        auth_weight = authority_scores.get(doc_id, 0.5)
        doc_scores[doc_id] += auth_weight / (k + rank + 1)
    
    # Dense contribution
    for rank, doc_id in enumerate(dense_results):
        auth_weight = authority_scores.get(doc_id, 0.5)
        doc_scores[doc_id] += auth_weight / (k + rank + 1)
    
    # Sort by combined score
    sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
    return [doc_id for doc_id, _ in sorted_docs]


# ============================================================
# AUTHORITY CONFLICT RESOLUTION
# ============================================================

@dataclass
class AuthorityConflict:
    """Conflict between two claims of different authority."""
    claim_a_id: str
    claim_b_id: str
    tier_a: AuthorityTier
    tier_b: AuthorityTier
    jurisdiction_a: str
    jurisdiction_b: str
    conflict_type: str
    claim_a_text: str = ""
    claim_b_text: str = ""
    metadata_a: Dict = field(default_factory=dict)
    metadata_b: Dict = field(default_factory=dict)


@dataclass
class ConflictResolution:
    """Result of conflict resolution."""
    conflict: AuthorityConflict
    prevailing_claim_id: Optional[str]
    overridden_claim_id: Optional[str]
    rule_applied: str
    rationale: str
    is_manual_review: bool


class AuthorityConflictDetector:
    """
    Detects conflicts between provisions of different authority tiers.
    """
    
    def detect(self, claims: List[ClaimAuthority]) -> List[AuthorityConflict]:
        conflicts = []
        
        # Group claims by legal proposition (simplified: by hierarchy path)
        claim_groups = self._group_by_proposition(claims)
        
        for group in claim_groups:
            if len(group) <= 1:
                continue
            
            # Check for contradictory claims
            for i, claim_a in enumerate(group):
                for claim_b in group[i+1:]:
                    if self._are_contradictory(claim_a, claim_b):
                        conflicts.append(AuthorityConflict(
                            claim_a_id=claim_a.claim_id,
                            claim_b_id=claim_b.claim_id,
                            tier_a=claim_a.composite_tier,
                            tier_b=claim_b.composite_tier,
                            jurisdiction_a=claim_a.metadata.get("jurisdiction", ""),
                            jurisdiction_b=claim_b.metadata.get("jurisdiction", ""),
                            conflict_type=self._classify_conflict(claim_a, claim_b),
                            metadata_a=claim_a.metadata,
                            metadata_b=claim_b.metadata
                        ))
        
        return conflicts
    
    def _group_by_proposition(self, claims: List[ClaimAuthority]) -> List[List[ClaimAuthority]]:
        """Group claims by legal proposition (simplified by hierarchy)."""
        groups = {}
        for claim in claims:
            # Use primary chunk's hierarchy as grouping key
            primary = claim.primary_chunk_authority
            if primary:
                key = f"{primary.source_id}:{primary.metadata.get('section', 'unknown')}"
            else:
                key = f"unknown:{claim.claim_id}"
            if key not in groups:
                groups[key] = []
            groups[key].append(claim)
        return list(groups.values())
    
    def _are_contradictory(self, claim_a: ClaimAuthority, claim_b: ClaimAuthority) -> bool:
        """Check if two claims are contradictory (simplified)."""
        # In reality, this would use NLI/entailment model
        # For now, assume different tiers on same provision = potential conflict
        return (claim_a.metadata.get("section") == claim_b.metadata.get("section") and
                claim_a.composite_tier != claim_b.composite_tier)
    
    def _classify_conflict(self, claim_a: ClaimAuthority, claim_b: ClaimAuthority) -> str:
        """Classify the type of conflict."""
        jur_a = claim_a.metadata.get("jurisdiction", "")
        jur_b = claim_b.metadata.get("jurisdiction", "")
        
        if jur_a == jur_b:
            if claim_a.composite_tier != claim_b.composite_tier:
                return "HIERARCHY"
            else:
                return "SAME_TIER_CONTRADICTION"
        
        # Different jurisdictions
        if self._is_federal_vs_state(claim_a, claim_b):
            return "FEDERAL_VS_STATE"
        if self._is_treaty_vs_domestic(claim_a, claim_b):
            return "TREATY_VS_DOMESTIC"
        
        return "CROSS_JURISDICTION"
    
    def _is_federal_vs_state(self, claim_a: ClaimAuthority, claim_b: ClaimAuthority) -> bool:
        """Check if conflict is federal vs state law."""
        # Simplified: check if one is central act, other is state act
        return False  # Would need jurisdiction metadata
    
    def _is_treaty_vs_domestic(self, claim_a: ClaimAuthority, claim_b: ClaimAuthority) -> bool:
        """Check if conflict is treaty vs domestic law."""
        return False


class AuthorityConflictResolver:
    """
    Resolves authority conflicts using legal hierarchy principles.
    """
    
    RESOLUTION_RULES = [
        # 1. Constitution > All
        ("CONSTITUTION_SUPREMACY",
         lambda c: c.conflict_type == "HIERARCHY" and 
                   (c.tier_a == AuthorityTier.TIER_1 and c.metadata_a.get("is_constitutional")) 
                   or (c.tier_b == AuthorityTier.TIER_1 and c.metadata_b.get("is_constitutional")),
         "Constitutional provision prevails"),
        
        # 2. Central Act > State Act (Union List)
        ("UNION_LIST_PREVAILS",
         lambda c: c.conflict_type == "FEDERAL_VS_STATE" and c.tier_a == c.tier_b == AuthorityTier.TIER_1
                   and c.metadata_a.get("list_entry") == "UNION",
         "Central legislation on Union List prevails (Art. 246)"),
        
        # 3. Lex posterior (later amendment)
        ("LEX_POSTERIOR",
         lambda c: c.conflict_type in ["HIERARCHY", "SAME_TIER_CONTRADICTION"] 
                   and c.metadata_a.get("effective_date") 
                   and c.metadata_b.get("effective_date")
                   and c.metadata_a["effective_date"] > c.metadata_b["effective_date"],
         "Later enactment prevails"),
        
        # 4. Lex specialis
        ("LEX_SPECIALIS",
         lambda c: c.metadata_a.get("specificity", 0) > c.metadata_b.get("specificity", 0),
         "More specific provision prevails"),
        
        # 5. Binding > Non-binding
        ("BINDING_PREVAILS",
         lambda c: c.tier_a.value <= 4 and c.tier_b.value >= 5,
         "Binding authority prevails over guidance"),
        
        # 6. Higher court > Lower court
        ("COURT_HIERARCHY",
         lambda c: c.tier_a == c.tier_b == AuthorityTier.TIER_3
                   and c.metadata_a.get("court_level") == "SUPREME"
                   and c.metadata_b.get("court_level") == "HIGH",
         "Supreme Court prevails over High Court"),
        
        # 7. Specialized tribunal > General court (on specialized matter)
        ("TRIBUNAL_SPECIALIZATION",
         lambda c: c.tier_a == AuthorityTier.TIER_4 and c.tier_b == AuthorityTier.TIER_3
                   and c.metadata_a.get("subject_matter") == c.metadata_a.get("tribunal_jurisdiction"),
         "Specialized tribunal prevails on its jurisdiction"),
    ]
    
    def resolve(self, conflict: AuthorityConflict) -> ConflictResolution:
        for rule_name, condition, rationale in self.RESOLUTION_RULES:
            if condition(conflict):
                # Determine winner based on rule
                winner_id, loser_id = self._determine_winner(conflict, rule_name)
                
                return ConflictResolution(
                    conflict=conflict,
                    prevailing_claim_id=winner_id,
                    overridden_claim_id=loser_id,
                    rule_applied=rule_name,
                    rationale=rationale,
                    is_manual_review=False
                )
        
        # No rule matched - escalate
        return ConflictResolution(
            conflict=conflict,
            prevailing_claim_id=None,
            overridden_claim_id=None,
            rule_applied="ESCALATE",
            rationale="No deterministic resolution rule matched",
            is_manual_review=True
        )
    
    def _determine_winner(self, conflict: AuthorityConflict, rule_name: str) -> tuple[Optional[str], Optional[str]]:
        """Determine winner based on rule."""
        # Default: higher tier (lower number) wins
        if conflict.tier_a.value < conflict.tier_b.value:
            return conflict.claim_a_id, conflict.claim_b_id
        elif conflict.tier_b.value < conflict.tier_a.value:
            return conflict.claim_b_id, conflict.claim_a_id
        
        # Same tier - use effective date (lex posterior)
        date_a = conflict.metadata_a.get("effective_date")
        date_b = conflict.metadata_b.get("effective_date")
        if date_a and date_b:
            if date_a > date_b:
                return conflict.claim_a_id, conflict.claim_b_id
            else:
                return conflict.claim_b_id, conflict.claim_a_id
        
        return conflict.claim_a_id, conflict.claim_b_id


# ============================================================
# CLAIM AUTHORITY COMPOSITION
# ============================================================

class ClaimAuthorityComposer:
    """
    Composes authority for generated claims from supporting evidence.
    """
    
    def __init__(self, authority_store: AuthorityStore):
        self.authority_store = authority_store
    
    def compose(self, claim_id: str, supporting_chunk_ids: List[str]) -> ClaimAuthority:
        """Compose authority for a claim from its supporting chunks."""
        
        # Get chunk authorities
        chunk_authorities = []
        for chunk_id in supporting_chunk_ids:
            auth = self.authority_store.get_chunk_authority(chunk_id)
            if auth:
                chunk_authorities.append(auth)
        
        if not chunk_authorities:
            # No authority info - use default
            return ClaimAuthority(
                claim_id=claim_id,
                primary_chunk_authority=None,
                supporting_chunk_authorities=[],
                composite_tier=AuthorityTier.TIER_6,
                composite_weight=0.5
            )
        
        # Primary = highest authority tier (lowest number)
        primary_auth = min(chunk_authorities, key=lambda ca: ca.authority_tier.value)
        
        # Composite tier = primary tier
        composite_tier = primary_auth.authority_tier
        
        # Composite weight = authority-weighted average
        total_weight = sum(ca.authority_weight for ca in chunk_authorities)
        weighted_sum = sum(ca.authority_weight * self._tier_weight(ca.authority_tier) for ca in chunk_authorities)
        composite_weight = weighted_sum / total_weight if total_weight > 0 else 0.5
        
        # Check for conflicts
        conflict_resolution = None
        if len(chunk_authorities) > 1:
            detector = AuthorityConflictDetector()
            # Would need to convert to ClaimAuthority objects for detection
            # Simplified for now
        
        return ClaimAuthority(
            claim_id=claim_id,
            primary_chunk_authority=primary_auth,
            supporting_chunk_authorities=chunk_authorities,
            composite_tier=composite_tier,
            composite_weight=composite_weight,
            conflict_resolution=conflict_resolution,
            metadata={
                "supporting_tiers": [ca.authority_tier.value for ca in chunk_authorities],
                "num_sources": len(chunk_authorities),
                "sources": [ca.source_id for ca in chunk_authorities]
            }
        )
    
    def _tier_weight(self, tier: AuthorityTier) -> float:
        return {1: 1.0, 2: 0.9, 3: 0.95, 4: 0.7, 5: 0.5, 6: 0.2}[tier.value]


# ============================================================
# AUTHORITY-AWARE CITATION GENERATION
# ============================================================

class AuthorityAwareCitationGenerator:
    """
    Generates citations with full authority metadata.
    """
    
    def generate(self, claim_authority: ClaimAuthority, chunk: LegalChunk) -> Citation:
        """Generate citation with authority metadata."""
        # Build citation text with authority indicators
        tier_labels = {1: "Act", 2: "Rule/Reg", 3: "Judgment", 4: "Order", 5: "Guideline", 6: "Commentary"}
        
        citation_text = f"{tier_labels.get(claim_authority.composite_tier.value, 'Source')}: "
        
        if chunk.hierarchy.section:
            citation_text += f"Section {chunk.hierarchy.section}"
        if chunk.hierarchy.act_short_title:
            citation_text += f", {chunk.hierarchy.act_short_title}"
        if chunk.effective_date:
            citation_text += f" (eff. {chunk.effective_date})"
        
        # Full citation
        full_citation = self._build_full_citation(chunk, claim_authority)
        
        return Citation(
            legal_citation=citation_text,
            source_reference=SourceReference(
                source_id=chunk.source_provenance.source_id,
                source_name=chunk.source_provenance.source_name,
                canonical_url=chunk.source_provenance.canonical_url,
                authority_tier=claim_authority.composite_tier,
                retrieved_at=chunk.source_provenance.acquired_at,
                content_hash=chunk.content_hash
            ),
            chunk_id=chunk.chunk_id,
            amendment_status=claim_authority.primary_chunk_authority.metadata.get("is_amended_version", False) and AmendmentStatus.AMENDED or AmendmentStatus.ORIGINAL,
            effective_date=chunk.effective_date,
            amendment_act=claim_authority.primary_chunk_authority.metadata.get("amendment_act")
        )
    
    def _build_full_citation(self, chunk: LegalChunk, auth: ClaimAuthority) -> str:
        parts = []
        
        if chunk.hierarchy.section:
            parts.append(f"Section {chunk.hierarchy.section}")
        if chunk.hierarchy.subsection:
            parts.append(f"Subsection {chunk.hierarchy.subsection}")
        if chunk.hierarchy.clause:
            parts.append(f"Clause {chunk.hierarchy.clause}")
        if chunk.hierarchy.act_short_title:
            parts.append(chunk.hierarchy.act_short_title)
        if chunk.hierarchy.act_year:
            parts.append(f"({chunk.hierarchy.act_year})")
        if chunk.source_provenance.gazette_number:
            parts.append(f"Gazette No. {chunk.source_provenance.gazette_number}")
        if chunk.source_provenance.gazette_date:
            parts.append(f"dated {chunk.source_provenance.gazette_date}")
        if chunk.effective_date:
            parts.append(f"effective {chunk.effective_date}")
        if chunk.source_provenance.canonical_url:
            parts.append(f"Available at: {chunk.source_provenance.canonical_url}")
        
        tier_desc = {1: "Primary Legislation", 2: "Subordinate Legislation", 
                     3: "Binding Judicial Precedent", 4: "Quasi-Judicial Order",
                     5: "Official Guidance", 6: "Secondary Source"}[auth.composite_tier.value]
        parts.append(f"[Authority Tier {auth.composite_tier.value}: {tier_desc}]")
        
        return "; ".join(parts)


# ============================================================
# AUTHORITY REGISTRY (KG-Backed)
# ============================================================

class AuthorityStore:
    """
    In-memory authority store (replace with KG-backed implementation).
    """
    
    def __init__(self):
        self.source_profiles: Dict[str, AuthorityProfile] = {}
        self.chunk_authorities: Dict[str, ChunkAuthority] = {}
        self.claim_authorities: Dict[str, ClaimAuthority] = {}
    
    def register_source(self, profile: AuthorityProfile):
        self.source_profiles[profile.source_id] = profile
    
    def get_source_authority(self, source_id: str) -> Optional[AuthorityProfile]:
        return self.source_profiles.get(source_id)
    
    def register_chunk_authority(self, auth: ChunkAuthority):
        self.chunk_authorities[auth.chunk_id] = auth
    
    def get_chunk_authority(self, chunk_id: str) -> Optional[ChunkAuthority]:
        return self.chunk_authorities.get(chunk_id)
    
    def register_claim_authority(self, auth: ClaimAuthority):
        self.claim_authorities[auth.claim_id] = auth
    
    def get_claim_authority(self, claim_id: str) -> Optional[ClaimAuthority]:
        return self.claim_authorities.get(claim_id)
    
    def get_authorities_for_chunks(self, chunk_ids: List[str]) -> List[ChunkAuthority]:
        return [self.chunk_authorities[cid] for cid in chunk_ids if cid in self.chunk_authorities]


# ============================================================
# AUTHORITY UPDATE PROCESSOR
# ============================================================

class AuthorityUpdateProcessor:
    """
    Processes amendments, new judgments, regulatory updates.
    """
    
    def __init__(self, classifier: AuthorityClassifier, store: AuthorityStore):
        self.classifier = classifier
        self.store = store
    
    def process_amendment(self, amendment_metadata: Dict, amended_source_id: str):
        """Update authority registry when amendment occurs."""
        # 1. Register amendment source
        amendment_profile = self.classifier.classify(amendment_metadata)
        self.store.register_source(amendment_profile)
        
        # 2. Update amended source
        amended_profile = self.store.get_source_authority(amended_source_id)
        if amended_profile:
            amended_profile.amendment_history.append(amendment_profile.source_id)
            amended_profile.superseded_by = amendment_profile.source_id
            self.store.register_source(amended_profile)  # Update
        
        # 3. Invalidate cached chunk authorities for amended source
        self._invalidate_chunk_cache(amended_source_id)
    
    def process_new_judgment(self, judgment_metadata: Dict):
        """Register new court judgment."""
        court = judgment_metadata.get("court", "")
        is_reported = judgment_metadata.get("is_reported", False)
        
        if court == "Supreme Court":
            tier = AuthorityTier.TIER_3
        elif court.startswith("High Court") and is_reported:
            tier = AuthorityTier.TIER_3
        elif court.startswith("High Court"):
            tier = AuthorityTier.TIER_4  # Unreported HC judgment
        else:
            tier = AuthorityTier.TIER_4  # Tribunal
        
        # Override classification for courts
        profile = AuthorityProfile(
            source_id=judgment_metadata["source_id"],
            source_type=SourceType.SUPREME_COURT_JUDGMENT if court == "Supreme Court" 
                       else SourceType.HIGH_COURT_JUDGMENT if court.startswith("High Court")
                       else SourceType.TRIBUNAL_ORDER,
            authority_tier=tier,
            jurisdiction_id=judgment_metadata.get("jurisdiction_id", "INDIA"),
            authority_name=court,
            enacting_authority=court,
            publication_date=judgment_metadata["date"] if isinstance(judgment_metadata["date"], date) else date.fromisoformat(judgment_metadata["date"]),
            is_binding=tier.value <= 3,
            metadata=judgment_metadata
        )
        
        self.store.register_source(profile)
        
        # Link to interpreted provisions
        for provision in judgment_metadata.get("interpreted_provisions", []):
            # Would create KG relationship: judgment INTERPRETS provision
            pass
    
    def _invalidate_chunk_cache(self, source_id: str):
        """Invalidate chunk authority cache for a source."""
        to_remove = [cid for cid, auth in self.store.chunk_authorities.items() 
                     if auth.source_id == source_id]
        for cid in to_remove:
            del self.store.chunk_authorities[cid]


# ============================================================
# AUTHORITY METRICS
# ============================================================

class AuthorityMetrics:
    """
    Tracks authority distribution across corpus, retrieval, generation.
    """
    
    def __init__(self, store: AuthorityStore):
        self.store = store
    
    def compute_corpus_distribution(self) -> Dict[int, int]:
        """Authority tier distribution in corpus."""
        from collections import Counter
        tiers = [p.authority_tier.value for p in self.store.source_profiles.values()]
        return dict(Counter(tiers))
    
    def compute_retrieval_authority(self, retrieved_chunks: List[str]) -> Dict:
        """Average authority tier of retrieved chunks."""
        tiers = []
        for chunk_id in retrieved_chunks:
            auth = self.store.get_chunk_authority(chunk_id)
            if auth:
                tiers.append(auth.authority_tier.value)
        
        if not tiers:
            return {"mean_tier": 0, "tier_distribution": {}, "binding_ratio": 0}
        
        from collections import Counter
        return {
            "mean_tier": sum(tiers) / len(tiers),
            "tier_distribution": dict(Counter(tiers)),
            "binding_ratio": sum(1 for t in tiers if t <= 4) / len(tiers)
        }
    
    def compute_generation_authority(self, claim_authorities: List[ClaimAuthority]) -> Dict:
        """Authority tier of claims in generated answers."""
        if not claim_authorities:
            return {"mean_tier": 0, "tier_distribution": {}, "binding_ratio": 0}
        
        from collections import Counter
        tiers = [ca.composite_tier.value for ca in claim_authorities]
        return {
            "mean_tier": sum(tiers) / len(tiers),
            "tier_distribution": dict(Counter(tiers)),
            "binding_ratio": sum(1 for t in tiers if t <= 4) / len(tiers)
        }
    
    def check_quality_gates(self, corpus_dist: Dict, retrieval_auth: Dict, generation_auth: Dict) -> List[str]:
        """Check quality gates, return list of violations."""
        violations = []
        
        # Corpus Tier 1+2 coverage > 60%
        total = sum(corpus_dist.values())
        tier12 = corpus_dist.get(1, 0) + corpus_dist.get(2, 0)
        if total > 0 and tier12 / total < 0.6:
            violations.append(f"Corpus Tier 1+2 coverage {tier12/total:.1%} < 60%")
        
        # Retrieval binding ratio > 80%
        if retrieval_auth.get("binding_ratio", 0) < 0.8:
            violations.append(f"Retrieval binding ratio {retrieval_auth.get('binding_ratio', 0):.1%} < 80%")
        
        # Generation binding ratio > 90%
        if generation_auth.get("binding_ratio", 0) < 0.9:
            violations.append(f"Generation binding ratio {generation_auth.get('binding_ratio', 0):.1%} < 90%")
        
        # Tier 6 in top-10 retrieval < 10%
        retrieval_tiers = retrieval_auth.get("tier_distribution", {})
        top10_total = sum(list(retrieval_tiers.values())[:10]) if retrieval_tiers else 0
        tier6_count = retrieval_tiers.get(6, 0)
        if top10_total > 0 and tier6_count / top10_total > 0.1:
            violations.append(f"Tier 6 in top retrieval {tier6_count/top10_total:.1%} > 10%")
        
        return violations


# ============================================================
# FACTORY FUNCTION
# ============================================================

def create_authority_system() -> tuple[AuthorityClassifier, AuthorityPropagator, AuthorityStore, AuthorityUpdateProcessor, AuthorityMetrics]:
    """Create all authority system components."""
    classifier = AuthorityClassifier()
    propagator = AuthorityPropagator(classifier)
    store = AuthorityStore()
    updater = AuthorityUpdateProcessor(classifier, store)
    metrics = AuthorityMetrics(store)
    return classifier, propagator, store, updater, metrics


# ============================================================
# WRAPPER CLASS FOR BACKWARD COMPATIBILITY
# ============================================================

class SourceAuthoritySystem:
    """Wrapper class for the authority system components."""
    
    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self.classifier = AuthorityClassifier()
        self.propagator = AuthorityPropagator(self.classifier)
        self.store = AuthorityStore()
        self.updater = AuthorityUpdateProcessor(self.classifier, self.store)
        self.metrics = AuthorityMetrics(self.store)
    
    def get_tier_score(self, tier: AuthorityTier) -> float:
        """Get weight score for authority tier."""
        return tier.weight
    
    def verify_tier(self, tier: AuthorityTier, metadata: Any) -> bool:
        """Verify authority tier is valid for metadata."""
        # Simple verification - check if tier is binding for given source type
        source_type = getattr(metadata, 'document_type', None) or getattr(metadata, 'source_type', None)
        if source_type:
            # Verify the tier matches expected tier for source type
            # Provide default source_id if not present
            classify_metadata = {"source_type": source_type}
            if hasattr(metadata, 'source_id'):
                classify_metadata["source_id"] = metadata.source_id
            elif hasattr(metadata, 'source_path'):
                classify_metadata["source_id"] = metadata.source_path
            else:
                classify_metadata["source_id"] = "unknown"
            
            # Add gazette_published for acts to be TIER_1
            if hasattr(source_type, 'value'):
                # It's an enum
                classify_metadata["source_type"] = source_type.value.lower()
            elif isinstance(source_type, str):
                classify_metadata["source_type"] = source_type.lower()
            
            # For acts/statutes, assume gazette_published if not specified
            custom_fields = getattr(metadata, "custom_fields", {}) or {}
            if classify_metadata["source_type"] in ["act", "amendment_act", "ordinance", "constitution", "treaty", "protocol", "cop_decision"]:
                classify_metadata["gazette_published"] = custom_fields.get("gazette_published", True)
            elif classify_metadata["source_type"] in ["rule", "regulation", "notification", "order"]:
                classify_metadata["gazette_published"] = custom_fields.get("gazette_published", False)
            
            expected_tier = self.classifier.classify(classify_metadata)
            return tier == expected_tier.authority_tier
        return True
    
    def classify_source(self, metadata: Dict[str, Any]) -> AuthorityTier:
        """Classify a source based on metadata."""
        return self.classifier.classify(metadata)
    
    def propagate_to_chunk(self, chunk: Any, source_tier: AuthorityTier) -> Any:
        """Propagate authority to chunk."""
        return self.propagator.propagate_to_chunk(chunk, source_tier)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get authority system statistics."""
        return self.metrics.get_corpus_authority_distribution()
