"""Maintained eligibility extension for pinned TreeQuest AB-MCTS-A.

Upstream: SakanaAI/treequest@3fac4f8b07e14ed92742471492a59764993b9ab0.
Extension v1 changes selection only: mask GEN actions by parent, and CONT arms
by reachable eligible descendants *before* TrialStore.create_trial. It retains
upstream posterior distributions, child indices, tell and backpropagation. No
ask/reject loop and no AB-MCTS-M rounded-score pruning are involved.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

from treequest.algos.ab_mcts_a.algo import ABMCTSA

TREEQUEST_REVISION = "3fac4f8b07e14ed92742471492a59764993b9ab0"
ELIGIBILITY_EXTENSION_VERSION = 1


class NoEligibleActions(RuntimeError):
    """The live tree has no permitted expansion, so no trial was created."""


@dataclass
class BranchDecision:
    scope: str
    target_id: Any
    state: str
    reason: str
    evidence: list[str] = field(default_factory=list)
    author: str = "search_policy"
    counters: dict[str, float] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def __post_init__(self):
        if self.scope not in {"candidate", "parent_action", "subtree", "strategy_family"}:
            raise ValueError(f"Unknown closure scope: {self.scope}")
        if self.state not in {"active", "suspended", "closed"}:
            raise ValueError(f"Unknown branch state: {self.state}")
        if not self.reason:
            raise ValueError("A branch decision requires a reason")
        if self.scope == "parent_action":
            if not isinstance(self.target_id, (list, tuple)) or len(self.target_id) != 2:
                raise ValueError("parent_action target must be (candidate_id, action)")
            self.target_id = list(self.target_id)

    def to_dict(self):
        return asdict(self)


@dataclass
class Eligibility:
    decisions: list[BranchDecision] = field(default_factory=list)
    action_families: dict[str, str] = field(default_factory=dict)

    @staticmethod
    def candidate_id(node) -> str:
        return str((node.state or {}).get("candidate_id", "baseline"))

    def active_decisions(self):
        latest = {}
        for decision in self.decisions:
            target = tuple(decision.target_id) if isinstance(decision.target_id, list) else decision.target_id
            latest[(decision.scope, target)] = decision
        return [d for d in latest.values() if d.state != "active"]

    def permits(self, node, action: str) -> bool:
        # Numerical invalidity differs from upstream scheduling INVALID.
        if (node.state or {}).get("valid") is False:
            return False
        candidate_id = self.candidate_id(node)
        ancestors = set()
        current = node
        while current is not None:
            ancestors.add(self.candidate_id(current))
            current = current.parent
        for decision in self.active_decisions():
            if decision.scope == "candidate" and decision.target_id == candidate_id:
                return False
            if decision.scope == "parent_action" and decision.target_id == [candidate_id, action]:
                return False
            if decision.scope == "subtree" and decision.target_id in ancestors:
                return False
            if (decision.scope == "strategy_family"
                    and decision.target_id == self.action_families.get(action, action)):
                return False
        return True


class EligibleABMCTSA(ABMCTSA):
    """AB-MCTS-A with a finite masked traversal; upstream ask/tell are retained."""

    def __init__(self, eligibility: Eligibility):
        super().__init__(model_selection_strategy="multiarm_bandit_thompson")
        self.eligibility = eligibility

    def _get_expand_node_and_action(self, state, actions):
        if not actions:
            raise NoEligibleActions("No allowed action")
        reachable = {}
        generation = {}
        # Node indices always increase from ancestors to descendants.
        for node in reversed(state.tree.get_nodes()):
            generation[node.expand_idx] = [a for a in actions if self.eligibility.permits(node, a)]
            reachable[node.expand_idx] = bool(generation[node.expand_idx]) or any(
                reachable[child.expand_idx] for child in node.children
            )
        node = state.tree.root
        if not reachable[node.expand_idx]:
            raise NoEligibleActions("No eligible parent/action remains")
        while True:
            # Always register the full stationary action set. Eligibility changes
            # mask distributions rather than discarding accumulated posteriors.
            posterior = state.thompson_states.get_or_create(node, list(state.all_rewards_store))
            allowed = generation[node.expand_idx]
            children = {
                index: posterior.node_probas["shared"][index]
                for index, child in enumerate(node.children)
                if reachable[child.expand_idx]
            }
            options = {}
            if allowed:
                options["GEN"] = posterior.gen_vs_cont_probas["shared"]["GEN"]
            if children:
                options["CONT"] = posterior.gen_vs_cont_probas["shared"]["CONT"]
            choice = posterior.thompson_sampling(options)
            if choice == "GEN":
                masked_rewards = {action: state.all_rewards_store[action] for action in allowed}
                action = posterior._select_best_action("thompson", masked_rewards)
                return node, action
            node = node.children[posterior.thompson_sampling(children)]
