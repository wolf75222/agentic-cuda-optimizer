"""Durable ask/tell adapter for genuine, pinned TreeQuest AB-MCTS-A."""
from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import json
import math
from pathlib import Path
import pickle
import threading
from functools import lru_cache

import numpy as np
import treequest
from treequest.trial import get_current_dt_str

from .experiment_store import ExperimentStore, canonical_json
from .treequest_eligibility import (
    BranchDecision, Eligibility, EligibleABMCTSA, NoEligibleActions,
    TREEQUEST_REVISION, ELIGIBILITY_EXTENSION_VERSION,
)

_RNG_LOCK = threading.RLock()


@lru_cache(maxsize=1)
def verify_treequest_pin():
    """Fail closed if upstream internals used by our extension have changed."""
    expected = {
        "algos/ab_mcts_a/algo.py":"1b2c70abd57694267a25213569d3ed5840815f026c334e8548c3885cb57ece54",
        "algos/ab_mcts_a/prob_state.py":"cb3c02bbe9fe715b3e8f2178a8986710fbe044b9ed8053fcc4cfa43351382654",
        "trial.py":"3af18abe318474aeef3cf62b0b6208bb3c808a0af755d4be88a3ed2d4a53c82c",
        "algos/tree.py":"76b043073ff8a2abfc943378d7f29c08ab1da88b9ca45ec0ba0f6ad4a3d39be4",
    }
    package = Path(treequest.__file__).parent
    for relative, expected_hash in expected.items():
        if hashlib.sha256((package/relative).read_bytes()).hexdigest() != expected_hash:
            raise RuntimeError(f"TreeQuest source differs from required {TREEQUEST_REVISION}: {relative}")


def stationary_reward(baseline_latencies, latencies, weights=None, *, valid=True):
    """Fixed-baseline weighted geometric gain, smoothly mapped to [0, 1]."""
    if not valid:
        return 0.0
    if not baseline_latencies or set(baseline_latencies) != set(latencies):
        raise ValueError("Latency case IDs must exactly match the frozen search suite")
    if weights is None:
        weights = {key:1/len(baseline_latencies) for key in baseline_latencies}
    if set(weights) != set(baseline_latencies) or any(not math.isfinite(w) or w <= 0 for w in weights.values()):
        raise ValueError("Each search case requires a finite positive weight")
    if not math.isclose(sum(weights.values()),1.0,rel_tol=0,abs_tol=1e-10):
        raise ValueError("Search weights must sum to one")
    log_gain = 0.0
    for key, baseline in baseline_latencies.items():
        measured = latencies[key]
        if any(not math.isfinite(value) or value <= 0 for value in (baseline,measured)):
            raise ValueError("Latencies must be finite and strictly positive")
        log_gain += weights[key]*(math.log(baseline)-math.log(measured))
    # Numerically stable logistic(log_gain), preserving sub-percent changes.
    if log_gain >= 0:
        return 1/(1+math.exp(-log_gain))
    value = math.exp(log_gain)
    return value/(1+value)


def reward_evaluation_evidence(result):
    """Keep measurement identity separate from mutable/per-trial bookkeeping.

    The complete trial result (including consumption) remains durable in trials
    and its result_ready event. A cached physical measurement may be linked by
    a second trial with a different cost, without changing its evidence bytes.
    """
    evidence = {key:value for key,value in result.items() if key not in {"consumption","trial_id"}}
    if "state" in evidence:
        evidence["state"] = {key:value for key,value in evidence["state"].items()
                             if key not in {"evaluations","current_validity"}}
    return evidence


class SearchPolicy:
    def __init__(self, store: ExperimentStore, run_id: str, baseline_state: dict,
                 baseline_latencies: dict, weights=None, actions=None,
                 contract_fingerprint="", environment_fingerprint="",
                 budget_limits=None, finalization_reserve=None, seed=0,
                 max_batch_size=8, action_families=None):
        verify_treequest_pin()
        self.store, self.run_id = store, run_id
        self.baseline_latencies = dict(baseline_latencies)
        if not self.baseline_latencies:
            raise ValueError("The frozen search performance suite cannot be empty")
        self.weights = weights if weights is not None else {key:1/len(baseline_latencies) for key in baseline_latencies}
        stationary_reward(self.baseline_latencies,self.baseline_latencies,self.weights)
        self.actions = list(actions if actions is not None else ["optimize"])
        if not self.actions or len(set(self.actions)) != len(self.actions) or any(not a for a in self.actions):
            raise ValueError("Actions must be unique nonempty names")
        if not isinstance(max_batch_size,int) or max_batch_size < 1:
            raise ValueError("max_batch_size must be a positive integer")
        self.max_batch_size = max_batch_size
        self.identity = {
            "contract":contract_fingerprint,"environment":environment_fingerprint,
            "baseline":hashlib.sha256(canonical_json(baseline_state).encode()).hexdigest(),
            "baseline_latencies":self.baseline_latencies,"weights":self.weights,
            "actions":self.actions,"action_families":action_families or {},
            "treequest_revision":TREEQUEST_REVISION,
            "eligibility_extension":ELIGIBILITY_EXTENSION_VERSION,
        }
        store.initialize_campaign(run_id,self.identity,budget_limits,finalization_reserve)
        store.archive_candidate(run_id,baseline_state)
        with store.transaction() as db:
            row = db.execute("SELECT snapshot FROM campaigns WHERE run_id=?",(run_id,)).fetchone()
            if row["snapshot"] is None:
                eligibility = Eligibility(action_families=action_families or {})
                tree = EligibleABMCTSA(eligibility).init_tree()
                # The virtual upstream root is the independently verified baseline.
                tree.tree.root.state = dict(baseline_state)
                tree.tree.root.score = 0.5
                snapshot = {"tree":tree,"eligibility":eligibility,
                            "rng":np.random.RandomState(seed).get_state()}
                self._save(db,snapshot)
                store._event(db,run_id,"campaign_initialized",self.identity)
        self.integrate_pending()

    def _load(self,db):
        row = db.execute("SELECT snapshot FROM campaigns WHERE run_id=?",(self.run_id,)).fetchone()
        return pickle.loads(row["snapshot"])

    def _save(self,db,snapshot):
        db.execute("UPDATE campaigns SET snapshot=?,tree_version=tree_version+1 WHERE run_id=?",
                   (pickle.dumps(snapshot,protocol=pickle.HIGHEST_PROTOCOL),self.run_id))

    @contextlib.contextmanager
    def _random_state(self,snapshot):
        # scipy's upstream distributions use NumPy's process-global RNG.
        # Isolate it and serialize sampling across campaign instances.
        with _RNG_LOCK:
            outer = np.random.get_state()
            np.random.set_state(snapshot["rng"])
            try:
                yield
                snapshot["rng"] = np.random.get_state()
            finally:
                np.random.set_state(outer)

    def ask(self, eligible_actions=None, budget_reservation=None):
        return self.ask_batch(1,eligible_actions,budget_reservation)[0]

    def ask_batch(self,batch_size,eligible_actions=None,budget_reservation=None):
        if not isinstance(batch_size,int) or not 1 <= batch_size <= self.max_batch_size:
            raise ValueError(f"batch_size must be within 1..{self.max_batch_size}")
        allowed = self.actions if eligible_actions is None else list(eligible_actions)
        if set(allowed)-set(self.actions):
            raise ValueError("Unknown action; the action set is frozen for a campaign")
        amount = dict(budget_reservation or {})
        if amount.get("trials",1) != 1:
            raise ValueError("Every TreeQuest trial must reserve exactly one trial")
        amount["trials"] = 1
        with self.store.transaction() as db:
            snapshot = self._load(db)
            state = snapshot["tree"]
            # The full action set is registered once, independent of masks.
            for action in self.actions:
                state.all_rewards_store.setdefault(action,[])
            algorithm = EligibleABMCTSA(snapshot["eligibility"])
            with self._random_state(snapshot):
                state,trials = algorithm.ask_batch(state,batch_size,allowed)
            for trial in trials:
                self.store._reserve(db,self.run_id,trial.trial_id,amount,"tree_trial",None)
                db.execute("INSERT INTO trials(run_id,trial_id,status,payload) VALUES (?,?,?,?)",
                           (self.run_id,trial.trial_id,"pending",canonical_json(dataclasses.asdict(trial))))
                self.store._event(db,self.run_id,"selection",dataclasses.asdict(trial))
            self._save(db,snapshot)
            return trials

    def assert_eligible(self,trial_id):
        with self.store.transaction() as db:
            snapshot = self._load(db)
            trial = snapshot["tree"].trial_store.running_trials.get(trial_id)
            if not trial:
                raise ValueError("Trial is not pending")
            node = snapshot["tree"].tree.get_node(trial.node_to_expand)
            if not snapshot["eligibility"].permits(node,trial.action):
                raise NoEligibleActions("Trial became inapplicable before launch")
            return trial

    def tell_once(self,trial_id,result,reward=None):
        """Persist evidence first, then integrate it transactionally exactly once.

        A successful result has evaluation_id, valid=True, latencies,
        purpose='tree_reward', fresh=True, and state (the exact selected local
        candidate dict). Incorrect kernels have valid=False and reward zero.
        Infrastructure failures use suspend_or_cancel, never synthetic rewards.
        """
        result = dict(result)
        if result.get("infrastructure_error"):
            raise ValueError("Infrastructure errors must be suspended/cancelled without reward")
        if "evaluation_id" not in result or "valid" not in result:
            raise ValueError("Result requires stable evaluation_id and explicit numerical validity")
        if not isinstance(result["valid"],bool) or not isinstance(result["evaluation_id"],str) or not result["evaluation_id"]:
            raise ValueError("Numerical validity must be boolean and evaluation_id a nonempty string")
        if result["valid"] and (result.get("purpose") != "tree_reward" or result.get("fresh") is not True):
            raise ValueError("Tree reward requires a new independent tree_reward measurement")
        computed = stationary_reward(self.baseline_latencies,result.get("latencies",{}),self.weights,valid=result["valid"])
        if reward is not None and not math.isclose(reward,computed,rel_tol=0,abs_tol=1e-12):
            raise ValueError("Reward differs from the frozen-baseline score")
        result["reward"] = computed
        state = dict(result.get("state",result.get("candidate",result)))
        state["valid"] = result["valid"]
        if "candidate_id" not in state:
            raise ValueError("Result requires an executable candidate identity")
        result["state"] = state
        payload = canonical_json(result)
        with self.store.transaction() as db:
            row = db.execute("SELECT status,result FROM trials WHERE run_id=? AND trial_id=?",(self.run_id,trial_id)).fetchone()
            if not row:
                raise KeyError(trial_id)
            if row["result"]:
                if row["result"] != payload:
                    raise ValueError("A trial result/reward cannot be rewritten; use a distinct measurement")
            elif row["status"] != "pending":
                # Archive late evidence, without reopening or rewarding a trial.
                self.store._add_evaluation(db,self.run_id,reward_evaluation_evidence(result))
                self.store._event(db,self.run_id,"late_result",{"trial_id":trial_id,"result":result})
                return False
            else:
                self.store._add_evaluation(db,self.run_id,reward_evaluation_evidence(result))
                db.execute("UPDATE trials SET result=?,evaluation_id=?,status='result_ready' WHERE run_id=? AND trial_id=?",
                           (payload,result["evaluation_id"],self.run_id,trial_id))
                db.execute("INSERT INTO integration_events VALUES (?,?,0)",(self.run_id,trial_id))
                self.store._event(db,self.run_id,"result_ready",{"trial_id":trial_id,"evaluation_id":result["evaluation_id"]})
        return self.integrate_pending(trial_id=trial_id)

    def _before_tree_commit(self, snapshot, trial_id):
        """Fault-injection seam: exceptions roll back tell and its event together."""

    def integrate_pending(self,trial_id=None):
        integrated = False
        while True:
            with self.store.transaction() as db:
                query = "SELECT t.* FROM integration_events e JOIN trials t USING(run_id,trial_id) WHERE e.run_id=? AND e.processed=0"
                params = [self.run_id]
                if trial_id:
                    query += " AND e.trial_id=?"
                    params.append(trial_id)
                row = db.execute(query+" LIMIT 1",params).fetchone()
                if row is None:
                    return integrated
                snapshot = self._load(db)
                state = snapshot["tree"]
                result = json.loads(row["result"])
                claim = db.execute("SELECT trial_id FROM reward_claims WHERE run_id=? AND evaluation_id=?",(self.run_id,row["evaluation_id"])).fetchone()
                if claim and claim["trial_id"] != row["trial_id"]:
                    self._cancel_in_snapshot(snapshot,row["trial_id"])
                    status = "duplicate"
                    self.store._event(db,self.run_id,"duplicate_measurement",{"trial_id":row["trial_id"],"existing_trial_id":claim["trial_id"],"evaluation_id":row["evaluation_id"]})
                else:
                    EligibleABMCTSA(snapshot["eligibility"]).tell(state,row["trial_id"],(result["state"],result["reward"]))
                    db.execute("INSERT OR IGNORE INTO reward_claims VALUES (?,?,?)",(self.run_id,row["evaluation_id"],row["trial_id"]))
                    status = "complete"
                    integrated = True
                    self.store._event(db,self.run_id,"tree_reward",{"trial_id":row["trial_id"],"evaluation_id":row["evaluation_id"],"reward":result["reward"]})
                amount = db.execute("SELECT amount FROM reservations WHERE run_id=? AND reservation_id=?",(self.run_id,row["trial_id"])).fetchone()
                actual = result.get("consumption",json.loads(amount["amount"]))
                self.store._reconcile(db,self.run_id,row["trial_id"],actual)
                self._before_tree_commit(snapshot,row["trial_id"])
                self._save(db,snapshot)
                db.execute("UPDATE integration_events SET processed=1 WHERE run_id=? AND trial_id=?",(self.run_id,row["trial_id"]))
                db.execute("UPDATE trials SET status=? WHERE run_id=? AND trial_id=?",(status,self.run_id,row["trial_id"]))

    @staticmethod
    def _cancel_in_snapshot(snapshot,trial_id):
        store = snapshot["tree"].trial_store
        trial = store.running_trials.pop(trial_id,None)
        if trial:
            store.finished_trials[trial_id] = dataclasses.replace(trial,trial_status="INVALID",completed_at=get_current_dt_str())

    def suspend_or_cancel(self,trial_id,reason,*,consumption=None):
        with self.store.transaction() as db:
            snapshot = self._load(db)
            if trial_id not in snapshot["tree"].trial_store.running_trials:
                return False
            self._cancel_in_snapshot(snapshot,trial_id)
            db.execute("UPDATE trials SET status='cancelled' WHERE run_id=? AND trial_id=?",(self.run_id,trial_id))
            db.execute("UPDATE integration_events SET processed=1 WHERE run_id=? AND trial_id=?",(self.run_id,trial_id))
            if consumption is not None:
                self.store._reconcile(db,self.run_id,trial_id,consumption)
            # Unknown physical execution cost retains its reservation until an
            # operator reconciles it; cancellation is not free GPU capacity.
            self.store._event(db,self.run_id,"trial_cancelled",{"trial_id":trial_id,"reason":reason})
            self._save(db,snapshot)
            return True

    def close(self,scope,target_id,reason,*,state="closed",evidence=None,author="search_policy",counters=None):
        decision = BranchDecision(scope,target_id,state,reason,evidence or [],author,counters or {})
        with self.store.transaction() as db:
            snapshot = self._load(db)
            snapshot["eligibility"].decisions.append(decision)
            self.store._event(db,self.run_id,"branch_decision",decision.to_dict())
            self._save(db,snapshot)
        return decision

    def set_candidate_validity(self,candidate_id,valid,reason):
        with self.store.transaction() as db:
            snapshot = self._load(db)
            for node in snapshot["tree"].tree.get_nodes():
                if Eligibility.candidate_id(node) == candidate_id:
                    node.state = dict(node.state,valid=valid)
            db.execute("INSERT OR REPLACE INTO candidate_status VALUES (?,?,?,?)",(self.run_id,candidate_id,int(valid),reason))
            self.store._event(db,self.run_id,"candidate_validity",{"candidate_id":candidate_id,"valid":valid,"reason":reason})
            self._save(db,snapshot)

    def pending_trials(self):
        with self.store.connection() as db:
            return list(self._load(db)["tree"].trial_store.running_trials.values())

    def snapshot(self):
        """Read-only independent snapshot for observability and tests."""
        with self.store.connection() as db:
            return self._load(db)
