"""Durable, genuine greedy-parent ablation for comparison with AB-MCTS-A.

This adapter creates no TreeQuest algorithm, tree, posterior or ask/tell state.
It selects the confirmed incumbent (initially the verified baseline), then
cycles through eligible strategy actions. It shares the campaign's fixed reward,
evaluation deduplication, transactional budgets and replay protocol. Numerical
failures never make a faster but unconfirmed candidate the next parent.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
import hashlib
import json
import math
import pickle
import uuid

from .experiment_store import canonical_json
from .search_policy import stationary_reward, reward_evaluation_evidence
from .treequest_eligibility import BranchDecision, NoEligibleActions


def _now():
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class GreedyTrial:
    trial_id: str
    node_to_expand: int
    action: str
    parent_state: dict
    score: float | None = None
    created_at: str = field(default_factory=_now)
    completed_at: str | None = None
    trial_status: str = "RUNNING"


class GreedySearchPolicy:
    """SearchPolicy-compatible control using confirmed-best + round robin."""

    def __init__(self,store,run_id,baseline_state,baseline_latencies,weights=None,actions=None,
                 contract_fingerprint="",environment_fingerprint="",budget_limits=None,
                 finalization_reserve=None,seed=0,max_batch_size=8,action_families=None):
        self.store,self.run_id = store,run_id
        self.baseline_id = str(baseline_state["candidate_id"])
        self.baseline_latencies = dict(baseline_latencies)
        if not self.baseline_latencies:
            raise ValueError("The frozen search suite cannot be empty")
        self.weights = weights if weights is not None else {key:1/len(baseline_latencies) for key in baseline_latencies}
        stationary_reward(self.baseline_latencies,self.baseline_latencies,self.weights)
        self.actions = list(actions if actions is not None else ["optimize"])
        if not self.actions or len(set(self.actions)) != len(self.actions) or any(not action for action in self.actions):
            raise ValueError("Actions must be unique nonempty names")
        if not isinstance(max_batch_size,int) or max_batch_size < 1:
            raise ValueError("max_batch_size must be a positive integer")
        self.max_batch_size = max_batch_size
        self.action_families = dict(action_families or {})
        self.identity = {
            "algorithm":"greedy_parent","algorithm_version":1,
            "parent_selection":"confirmed_incumbent","action_selection":"round_robin",
            "contract":contract_fingerprint,"environment":environment_fingerprint,
            "baseline":hashlib.sha256(canonical_json(baseline_state).encode()).hexdigest(),
            "baseline_latencies":self.baseline_latencies,"weights":self.weights,
            "actions":self.actions,"action_families":self.action_families,
        }
        store.initialize_campaign(run_id,self.identity,budget_limits,finalization_reserve)
        store.archive_candidate(run_id,baseline_state)
        with store.transaction() as db:
            row = db.execute("SELECT snapshot FROM campaigns WHERE run_id=?",(run_id,)).fetchone()
            if row["snapshot"] is None:
                self._save(db,{
                    "algorithm":"greedy_parent","version":1,"cursor":0,
                    "candidates":{self.baseline_id:dict(baseline_state)},
                    "parents":{self.baseline_id:None},"indices":{self.baseline_id:-1},
                    "pending":{},"finished":{},"observations":{},"decisions":[],
                })
                store._event(db,run_id,"campaign_initialized",self.identity)
        self.integrate_pending()

    def _load(self,db):
        row = db.execute("SELECT snapshot FROM campaigns WHERE run_id=?",(self.run_id,)).fetchone()
        return pickle.loads(row["snapshot"])

    def _save(self,db,snapshot):
        db.execute("UPDATE campaigns SET snapshot=?,tree_version=tree_version+1 WHERE run_id=?",
                   (pickle.dumps(snapshot,protocol=pickle.HIGHEST_PROTOCOL),self.run_id))

    def _candidate(self,db,snapshot,candidate_id):
        if candidate_id not in snapshot["candidates"]:
            row = db.execute("SELECT payload FROM candidates WHERE run_id=? AND candidate_id=?",(self.run_id,candidate_id)).fetchone()
            if not row:
                raise KeyError(candidate_id)
            candidate = json.loads(row["payload"])
            snapshot["candidates"][candidate_id] = candidate
            snapshot["parents"][candidate_id] = candidate.get("tree_parent_id",self.baseline_id)
            snapshot["indices"][candidate_id] = len(snapshot["indices"])-1
        return snapshot["candidates"][candidate_id]

    def _permits(self,db,snapshot,parent,action):
        candidate_id = str(parent["candidate_id"])
        validity = db.execute("SELECT valid FROM candidate_status WHERE run_id=? AND candidate_id=?",(self.run_id,candidate_id)).fetchone()
        if parent.get("valid") is False or (validity is not None and validity["valid"] == 0):
            return False
        ancestors = set()
        current = candidate_id
        while current is not None and current not in ancestors:
            ancestors.add(current)
            current = snapshot["parents"].get(current)
        latest = {}
        for decision in snapshot["decisions"]:
            target = tuple(decision.target_id) if isinstance(decision.target_id,list) else decision.target_id
            latest[(decision.scope,target)] = decision
        for decision in latest.values():
            if decision.state == "active":
                continue
            if decision.scope == "candidate" and decision.target_id == candidate_id:
                return False
            if decision.scope == "parent_action" and decision.target_id == [candidate_id,action]:
                return False
            if decision.scope == "subtree" and decision.target_id in ancestors:
                return False
            if decision.scope == "strategy_family" and decision.target_id == self.action_families.get(action,action):
                return False
        return True

    def ask(self,eligible_actions=None,budget_reservation=None):
        return self.ask_batch(1,eligible_actions,budget_reservation)[0]

    def ask_batch(self,batch_size,eligible_actions=None,budget_reservation=None):
        if not isinstance(batch_size,int) or not 1 <= batch_size <= self.max_batch_size:
            raise ValueError(f"batch_size must be within 1..{self.max_batch_size}")
        allowed = set(self.actions if eligible_actions is None else eligible_actions)
        if allowed-set(self.actions):
            raise ValueError("Unknown action; the action set is frozen")
        amount = dict(budget_reservation or {})
        if amount.get("trials",1) != 1:
            raise ValueError("Every search trial must reserve exactly one trial")
        amount["trials"] = 1
        with self.store.transaction() as db:
            snapshot = self._load(db)
            row = db.execute("SELECT incumbent,incumbent_version FROM campaigns WHERE run_id=?",(self.run_id,)).fetchone()
            candidate_id = row["incumbent"] or self.baseline_id
            parent = self._candidate(db,snapshot,candidate_id)
            eligible = {action for action in self.actions if action in allowed and self._permits(db,snapshot,parent,action)}
            if not eligible:
                raise NoEligibleActions("The confirmed greedy incumbent has no eligible action")
            trials = []
            for _ in range(batch_size):
                # At most len(actions) steps, with no rejected trial creation.
                for offset in range(len(self.actions)):
                    index = (snapshot["cursor"]+offset)%len(self.actions)
                    if self.actions[index] in eligible:
                        action = self.actions[index]
                        snapshot["cursor"] = (index+1)%len(self.actions)
                        break
                trial = GreedyTrial(uuid.uuid4().hex,snapshot["indices"][candidate_id],action,dict(parent))
                self.store._reserve(db,self.run_id,trial.trial_id,amount,"greedy_trial",None)
                snapshot["pending"][trial.trial_id] = trial
                db.execute("INSERT INTO trials(run_id,trial_id,status,payload) VALUES (?,?,?,?)",
                    (self.run_id,trial.trial_id,"pending",canonical_json(asdict(trial))))
                self.store._event(db,self.run_id,"selection",dict(asdict(trial),
                    search_algorithm="greedy_parent",incumbent_version=row["incumbent_version"]))
                trials.append(trial)
            self._save(db,snapshot)
            return trials

    def assert_eligible(self,trial_id):
        with self.store.transaction() as db:
            snapshot = self._load(db)
            trial = snapshot["pending"].get(trial_id)
            if trial is None:
                raise ValueError("Trial is not pending")
            if not self._permits(db,snapshot,trial.parent_state,trial.action):
                raise NoEligibleActions("Trial became inapplicable before launch")
            return trial

    def tell_once(self,trial_id,result,reward=None):
        result = dict(result)
        if result.get("infrastructure_error"):
            raise ValueError("Infrastructure errors cannot receive a performance reward")
        if not isinstance(result.get("valid"),bool) or not isinstance(result.get("evaluation_id"),str) or not result["evaluation_id"]:
            raise ValueError("Result requires explicit validity and nonempty evaluation_id")
        if result["valid"] and (result.get("purpose") != "tree_reward" or result.get("fresh") is not True):
            raise ValueError("The shared reward requires an independent tree_reward measurement")
        computed = stationary_reward(self.baseline_latencies,result.get("latencies",{}),self.weights,valid=result["valid"])
        if reward is not None and not math.isclose(reward,computed,rel_tol=0,abs_tol=1e-12):
            raise ValueError("Reward differs from the frozen-baseline score")
        state = dict(result.get("state",result.get("candidate",result)))
        if "candidate_id" not in state:
            raise ValueError("Result requires an executable candidate identity")
        state["valid"] = result["valid"]
        result.update(state=state,reward=computed)
        payload = canonical_json(result)
        with self.store.transaction() as db:
            row = db.execute("SELECT status,result FROM trials WHERE run_id=? AND trial_id=?",(self.run_id,trial_id)).fetchone()
            if row is None:
                raise KeyError(trial_id)
            if row["result"]:
                if row["result"] != payload:
                    raise ValueError("A trial result/reward cannot be rewritten")
            elif row["status"] != "pending":
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

    def _before_tree_commit(self,snapshot,trial_id):
        """Compatibility fault-injection seam before the durable ledger commit."""

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
                result = json.loads(row["result"])
                trial = snapshot["pending"].pop(row["trial_id"])
                claim = db.execute("SELECT trial_id FROM reward_claims WHERE run_id=? AND evaluation_id=?",(self.run_id,row["evaluation_id"])).fetchone()
                duplicate = claim is not None and claim["trial_id"] != row["trial_id"]
                if duplicate:
                    status = "duplicate"
                    snapshot["finished"][trial.trial_id] = replace(trial,trial_status="INVALID",completed_at=_now())
                    self.store._event(db,self.run_id,"duplicate_measurement",{"trial_id":trial.trial_id,"existing_trial_id":claim["trial_id"],"evaluation_id":row["evaluation_id"]})
                else:
                    candidate_id = str(result["state"]["candidate_id"])
                    snapshot["candidates"][candidate_id] = result["state"]
                    snapshot["parents"][candidate_id] = str(trial.parent_state["candidate_id"])
                    snapshot["indices"].setdefault(candidate_id,len(snapshot["indices"])-1)
                    snapshot["observations"][trial.trial_id] = {"candidate_id":candidate_id,"evaluation_id":row["evaluation_id"],"reward":result["reward"]}
                    snapshot["finished"][trial.trial_id] = replace(trial,score=result["reward"],trial_status="COMPLETE",completed_at=_now())
                    db.execute("INSERT OR IGNORE INTO reward_claims VALUES (?,?,?)",(self.run_id,row["evaluation_id"],trial.trial_id))
                    self.store._event(db,self.run_id,"tree_reward",dict(snapshot["observations"][trial.trial_id],trial_id=trial.trial_id,search_algorithm="greedy_parent"))
                    status = "complete"
                    integrated = True
                reservation = db.execute("SELECT amount FROM reservations WHERE run_id=? AND reservation_id=?",(self.run_id,trial.trial_id)).fetchone()
                self.store._reconcile(db,self.run_id,trial.trial_id,result.get("consumption",json.loads(reservation["amount"])))
                self._before_tree_commit(snapshot,trial.trial_id)
                self._save(db,snapshot)
                db.execute("UPDATE trials SET status=? WHERE run_id=? AND trial_id=?",(status,self.run_id,trial.trial_id))
                db.execute("UPDATE integration_events SET processed=1 WHERE run_id=? AND trial_id=?",(self.run_id,trial.trial_id))

    def suspend_or_cancel(self,trial_id,reason,*,consumption=None):
        with self.store.transaction() as db:
            snapshot = self._load(db)
            trial = snapshot["pending"].pop(trial_id,None)
            if trial is None:
                return False
            snapshot["finished"][trial_id] = replace(trial,trial_status="INVALID",completed_at=_now())
            db.execute("UPDATE trials SET status='cancelled' WHERE run_id=? AND trial_id=?",(self.run_id,trial_id))
            db.execute("UPDATE integration_events SET processed=1 WHERE run_id=? AND trial_id=?",(self.run_id,trial_id))
            if consumption is not None:
                self.store._reconcile(db,self.run_id,trial_id,consumption)
            self.store._event(db,self.run_id,"trial_cancelled",{"trial_id":trial_id,"reason":reason})
            self._save(db,snapshot)
            return True

    def close(self,scope,target_id,reason,*,state="closed",evidence=None,author="search_policy",counters=None):
        decision = BranchDecision(scope,target_id,state,reason,evidence or [],author,counters or {})
        with self.store.transaction() as db:
            snapshot = self._load(db)
            snapshot["decisions"].append(decision)
            self.store._event(db,self.run_id,"branch_decision",decision.to_dict())
            self._save(db,snapshot)
        return decision

    def set_candidate_validity(self,candidate_id,valid,reason):
        with self.store.transaction() as db:
            snapshot = self._load(db)
            if candidate_id in snapshot["candidates"]:
                snapshot["candidates"][candidate_id] = dict(snapshot["candidates"][candidate_id],valid=valid)
            db.execute("INSERT OR REPLACE INTO candidate_status VALUES (?,?,?,?)",(self.run_id,candidate_id,int(valid),reason))
            self.store._event(db,self.run_id,"candidate_validity",{"candidate_id":candidate_id,"valid":valid,"reason":reason})
            self._save(db,snapshot)

    def pending_trials(self):
        with self.store.connection() as db:
            return list(self._load(db)["pending"].values())

    def snapshot(self):
        with self.store.connection() as db:
            return self._load(db)
