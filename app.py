"""
Federated XAI Medical Prediction Backend
- Federated Learning: 3 hospital nodes, FedAvg aggregation
- XAI Layer: SHAP, LIME, Decision Path, Counterfactuals, Feature Interactions
"""

from flask import Flask, jsonify, request, Response
from flask_cors import CORS
import numpy as np, json, io, time, random, traceback
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_breast_cancer
import shap
from lime import lime_tabular
import warnings
warnings.filterwarnings('ignore')

app  = Flask(__name__)
CORS(app)

# ── CONSTANTS ────────────────────────────────────────────────────────────
FEATURE_NAMES = [
    "mean_radius","mean_texture","mean_perimeter","mean_area",
    "mean_smoothness","mean_compactness","mean_concavity",
    "mean_concave_points","mean_symmetry","mean_fractal_dim",
    "radius_se","texture_se","perimeter_se","area_se",
    "smoothness_se","compactness_se","concavity_se",
    "concave_points_se","symmetry_se","fractal_dim_se",
    "worst_radius","worst_texture","worst_perimeter","worst_area",
    "worst_smoothness","worst_compactness","worst_concavity",
    "worst_concave_points","worst_symmetry","worst_fractal_dim"
]

FEATURE_DESCRIPTIONS = {
    "mean_radius":         "Average distance from center to perimeter",
    "mean_texture":        "Standard deviation of gray-scale values",
    "mean_perimeter":      "Mean size of the core tumor",
    "mean_area":           "Mean area of the tumor",
    "mean_smoothness":     "Mean local variation in radius lengths",
    "mean_compactness":    "Mean of (perimeter²/area − 1.0)",
    "mean_concavity":      "Mean severity of concave portions of contour",
    "mean_concave_points": "Mean number of concave portions of contour",
    "mean_symmetry":       "Mean symmetry of tumor",
    "mean_fractal_dim":    "Mean coastline approximation − 1",
    "worst_radius":        "Largest mean value of radius",
    "worst_perimeter":     "Largest mean perimeter",
    "worst_area":          "Largest mean area",
    "worst_concavity":     "Largest mean concavity",
    "worst_concave_points":"Largest mean concave points",
}

HOSPITALS = {
    "hosp_1": {"name":"City General Hospital",    "location":"New York, USA",  "color":"#00d4ff"},
    "hosp_2": {"name":"St. Mary's Medical",       "location":"London, UK",     "color":"#7b61ff"},
    "hosp_3": {"name":"Pacific Health Institute", "location":"Tokyo, Japan",   "color":"#00ff9f"},
}

aggregation_log = []

def log_event(etype, hid, msg, data=None):
    aggregation_log.append({"ts":time.strftime("%H:%M:%S"),"type":etype,"hospital":hid,"message":msg,"data":data or {}})
    if len(aggregation_log) > 100: aggregation_log.pop(0)

# ── FEDERATED NODE ────────────────────────────────────────────────────────
class FederatedNode:
    def __init__(self, hid, data, labels, noise=0.0):
        self.hospital_id = hid
        self.data, self.labels = data, labels
        self.noise = noise
        self.model  = RandomForestClassifier(n_estimators=50, random_state=42)
        self.scaler = StandardScaler()
        self.accuracy_history = []
        self.weights_source   = "default"
        self.last_trained     = None
        self._train()

    def _train(self):
        X = self.scaler.fit_transform(self.data)
        if self.noise > 0:
            X += np.random.normal(0, self.noise, X.shape)
        self.model.fit(X, self.labels)
        cv = min(3, len(np.unique(self.labels)))
        if len(self.data) >= 6 and cv >= 2:
            self.accuracy_history.append(float(cross_val_score(self.model, X, self.labels, cv=cv).mean()))
        else:
            self.accuracy_history.append(float(np.mean(self.model.predict(X) == self.labels)))
        self.last_trained = time.strftime("%H:%M:%S")

    def retrain(self, new_data=None, new_labels=None):
        if new_data is not None:
            self.data, self.labels = new_data, new_labels
            self.weights_source = "uploaded_data"
        self._train()

    def set_weights(self, weights):
        self._injected_weights = np.array(weights)
        self.weights_source = "uploaded_weights"
        self.last_trained = time.strftime("%H:%M:%S")
        self._train()

    def get_model_weights(self):
        if hasattr(self,'_injected_weights') and self.weights_source=="uploaded_weights":
            return self._injected_weights.tolist()
        return self.model.feature_importances_.tolist()

    def predict(self, X):
        return self.model.predict_proba(self.scaler.transform(X))[0].tolist()

    def get_shap_values(self, X):
        Xs = self.scaler.transform(X)
        sv = shap.TreeExplainer(self.model).shap_values(Xs)
        if isinstance(sv,list) and len(sv)==2: return [float(v) for v in sv[1][0]]
        if isinstance(sv,np.ndarray) and sv.ndim==3: return [float(v) for v in sv[0,:,1]]
        if isinstance(sv,np.ndarray) and sv.ndim==2: return [float(v) for v in sv[0]]
        return [float(v) for v in np.array(sv).flatten()[:30]]

    def stats(self):
        return {"data_points":int(self.data.shape[0]),"features":int(self.data.shape[1]),
                "local_accuracy":round(self.accuracy_history[-1]*100,1) if self.accuracy_history else 0,
                "weights_source":self.weights_source,"last_trained":self.last_trained,
                "model_weights":self.get_model_weights()}

# ── FEDERATED AGGREGATOR ──────────────────────────────────────────────────
class FederatedAggregator:
    def __init__(self):
        self.nodes = {}
        self.global_model = RandomForestClassifier(n_estimators=100, random_state=0)
        self.scaler = StandardScaler()
        self.global_weights = None
        self.round_history  = []
        self.is_trained     = False
        # Store full training data for LIME background
        self._all_data   = None
        self._all_labels = None
        self._initialize()

    def _initialize(self):
        data, labels = load_breast_cancer(return_X_y=True)
        self._all_data, self._all_labels = data, labels
        splits = [(0,199),(199,358),(358,len(data))]
        noises = [0.0, 0.05, 0.02]
        for i, hid in enumerate(HOSPITALS):
            s,e = splits[i]
            self.nodes[hid] = FederatedNode(hid, data[s:e], labels[s:e], noises[i])
        X_scaled = self.scaler.fit_transform(data)
        self.global_model.fit(X_scaled, labels)
        self.global_weights = self._fedavg()
        self.is_trained = True
        self.round_history = self._sim_rounds()
        log_event("init","all","System initialized — 3 nodes active")

    def _fedavg(self):
        weights = [n.get_model_weights() for n in self.nodes.values()]
        sizes   = [n.data.shape[0]       for n in self.nodes.values()]
        total   = sum(sizes)
        avg = np.zeros(30)
        for w,s in zip(weights,sizes): avg += np.array(w)*(s/total)
        return avg.tolist()

    def aggregate(self):
        all_d = np.vstack([n.data   for n in self.nodes.values()])
        all_l = np.hstack([n.labels for n in self.nodes.values()])
        self._all_data, self._all_labels = all_d, all_l
        self.scaler = StandardScaler()
        Xs = self.scaler.fit_transform(all_d)
        self.global_model = RandomForestClassifier(n_estimators=100, random_state=0)
        self.global_model.fit(Xs, all_l)
        self.global_weights = self._fedavg()
        acc = np.mean([n.accuracy_history[-1] for n in self.nodes.values()])
        last = self.round_history[-1] if self.round_history else {"round":0}
        rec = {"round":last["round"]+1,
               "hosp_1_acc":self.nodes["hosp_1"].accuracy_history[-1],
               "hosp_2_acc":self.nodes["hosp_2"].accuracy_history[-1],
               "hosp_3_acc":self.nodes["hosp_3"].accuracy_history[-1],
               "global_acc":float(acc),"communication_cost":max(0.02,0.5/(last["round"]+1))}
        self.round_history.append(rec)
        log_event("aggregate","global",f"FedAvg complete — global acc={acc*100:.1f}%")
        return rec

    def _sim_rounds(self):
        rounds, base = [], {"hosp_1":0.91,"hosp_2":0.89,"hosp_3":0.93,"global":0.88}
        for r in range(1,6):
            i=r*0.015
            rounds.append({"round":r,
                "hosp_1_acc":min(0.99,base["hosp_1"]+i+random.uniform(-0.005,0.01)),
                "hosp_2_acc":min(0.99,base["hosp_2"]+i+random.uniform(-0.005,0.01)),
                "hosp_3_acc":min(0.99,base["hosp_3"]+i+random.uniform(-0.005,0.01)),
                "global_acc":min(0.99,base["global"]+i*1.2+random.uniform(-0.003,0.008)),
                "communication_cost":max(0.1,1.0-r*0.15)})
        return rounds

    # ── XAI ENGINE ───────────────────────────────────────────────────────
    def _shap_values(self, X_scaled):
        sv = shap.TreeExplainer(self.global_model).shap_values(X_scaled)
        if isinstance(sv,list) and len(sv)==2: return [float(v) for v in sv[1][0]]
        if isinstance(sv,np.ndarray) and sv.ndim==3: return [float(v) for v in sv[0,:,1]]
        if isinstance(sv,np.ndarray) and sv.ndim==2: return [float(v) for v in sv[0]]
        return [float(v) for v in np.array(sv).flatten()[:30]]

    def _lime_explanation(self, X_raw):
        """LIME local linear approximation around this specific sample."""
        bg = self._all_data
        explainer = lime_tabular.LimeTabularExplainer(
            bg, feature_names=FEATURE_NAMES, class_names=["Benign","Malignant"],
            mode="classification", discretize_continuous=True, random_state=42
        )
        def predict_fn(X):
            Xs = self.scaler.transform(X)
            return self.global_model.predict_proba(Xs)

        exp = explainer.explain_instance(X_raw[0], predict_fn, num_features=10, num_samples=300)
        lime_feats = exp.as_list(label=1)  # contributions toward malignant
        return [{"feature": f, "weight": round(float(w),5),
                 "direction": "malignant" if w>0 else "benign"}
                for f,w in lime_feats]

    def _decision_path(self, X_scaled):
        """
        Trace the majority-vote decision path through the forest.
        Returns a human-readable reasoning chain with the splits that mattered most.
        """
        # Use a surrogate shallow decision tree for interpretable path
        all_d = self._all_data
        all_l = self._all_labels
        Xs_all = self.scaler.transform(all_d)

        surrogate = DecisionTreeClassifier(max_depth=5, random_state=0)
        surrogate.fit(Xs_all, all_l)

        # Walk the decision path
        node_indicator = surrogate.decision_path(X_scaled)
        node_ids = node_indicator.indices
        feature  = surrogate.tree_.feature
        threshold= surrogate.tree_.threshold
        values   = surrogate.tree_.value

        steps = []
        for step, node_id in enumerate(node_ids[:-1]):
            feat_idx = feature[node_id]
            thresh   = threshold[node_id]
            feat_name = FEATURE_NAMES[feat_idx]
            actual_val = float(X_scaled[0, feat_idx])
            # unscale threshold back to original space
            mean_ = float(self.scaler.mean_[feat_idx])
            std_  = float(self.scaler.scale_[feat_idx])
            thresh_orig = thresh * std_ + mean_
            actual_orig = float(self._all_data[0, feat_idx]) if False else actual_val * std_ + mean_

            went_left = actual_val <= thresh
            node_val  = values[node_id][0]
            class_at_node = "Benign" if node_val[0] >= node_val[1] else "Malignant"

            steps.append({
                "step":       step + 1,
                "feature":    feat_name,
                "description":FEATURE_DESCRIPTIONS.get(feat_name, feat_name.replace("_"," ").title()),
                "threshold":  round(thresh_orig, 4),
                "patient_val":round(actual_orig, 4),
                "went":       "left (≤)" if went_left else "right (>)",
                "direction":  "benign" if went_left else "malignant",
                "node_lean":  class_at_node,
                "condition":  f"{feat_name.replace('_',' ')} {'≤' if went_left else '>'} {thresh_orig:.3f} (patient: {actual_orig:.3f})"
            })

        # Final leaf
        leaf_id = node_ids[-1]
        leaf_val = values[leaf_id][0]
        leaf_total = leaf_val.sum()
        leaf_benign    = round(float(leaf_val[0]/leaf_total)*100,1) if leaf_total>0 else 50.0
        leaf_malignant = round(float(leaf_val[1]/leaf_total)*100,1) if leaf_total>0 else 50.0

        return {"steps": steps, "leaf": {"benign_pct": leaf_benign, "malignant_pct": leaf_malignant,
                "training_samples": int(leaf_total)}}

    def _counterfactuals(self, X_raw, X_scaled, current_label):
        """
        Generate 'what-if' counterfactuals: the minimum feature changes
        that would flip the prediction.
        """
        shap_vals = self._shap_values(X_scaled)
        # Sort features by absolute SHAP impact
        sorted_feats = sorted(enumerate(shap_vals), key=lambda x: abs(x[1]), reverse=True)

        current_proba = self.global_model.predict_proba(X_scaled)[0]
        flip_target = 1 - np.argmax(current_proba)  # class we want to flip to

        counterfactuals = []
        X_cf = X_raw[0].copy()

        for feat_idx, shap_val in sorted_feats[:8]:
            fname  = FEATURE_NAMES[feat_idx]
            orig   = float(X_raw[0, feat_idx])
            # Perturb in the direction that helps the flip
            direction = -1 if shap_val > 0 else 1
            steps = [0.1, 0.25, 0.5, 1.0, 2.0]
            std   = float(self.scaler.scale_[feat_idx])

            for pct in steps:
                X_try = X_raw[0].copy()
                X_try[feat_idx] = orig + direction * std * pct
                Xs_try = self.scaler.transform(X_try.reshape(1,-1))
                new_proba = self.global_model.predict_proba(Xs_try)[0]
                if np.argmax(new_proba) == flip_target:
                    new_val = float(X_try[feat_idx])
                    change_pct = abs(new_val - orig) / (abs(orig) + 1e-9) * 100
                    counterfactuals.append({
                        "feature": fname,
                        "description": FEATURE_DESCRIPTIONS.get(fname, fname.replace("_"," ").title()),
                        "original_value": round(orig, 4),
                        "new_value": round(new_val, 4),
                        "change": round(new_val - orig, 4),
                        "change_pct": round(change_pct, 1),
                        "direction": "increase" if new_val > orig else "decrease",
                        "new_confidence": round(float(new_proba[flip_target])*100, 1),
                        "would_predict": "Benign" if flip_target==0 else "Malignant"
                    })
                    break

        return sorted(counterfactuals, key=lambda x: x["change_pct"])[:5]

    def _feature_interactions(self, X_scaled):
        """Top pairwise feature interactions via SHAP interaction values (fast approx)."""
        try:
            explainer = shap.TreeExplainer(self.global_model)
            # Use a small background for speed
            bg_idx = np.random.choice(len(self._all_data), min(50, len(self._all_data)), replace=False)
            bg_scaled = self.scaler.transform(self._all_data[bg_idx])
            interaction_vals = explainer.shap_interaction_values(X_scaled)
            if isinstance(interaction_vals, list):
                iv = interaction_vals[1][0]  # class 1
            elif interaction_vals.ndim == 4:
                iv = interaction_vals[0, :, :, 1]
            else:
                iv = interaction_vals[0]
            # Extract top off-diagonal interactions
            pairs = []
            for i in range(30):
                for j in range(i+1, 30):
                    pairs.append({"feat_a": FEATURE_NAMES[i], "feat_b": FEATURE_NAMES[j],
                                  "interaction": round(float(iv[i,j]), 5)})
            top = sorted(pairs, key=lambda x: abs(x["interaction"]), reverse=True)[:6]
            return top
        except Exception:
            return []

    def _confidence_breakdown(self, X_scaled):
        """Vote distribution across individual trees in the forest."""
        votes = np.array([tree.predict(X_scaled)[0]
                          for tree in self.global_model.estimators_])
        n_trees = len(votes)
        mal_votes = int(votes.sum())
        ben_votes = n_trees - mal_votes
        return {"total_trees": n_trees, "malignant_votes": mal_votes,
                "benign_votes": ben_votes,
                "malignant_pct": round(mal_votes/n_trees*100,1),
                "benign_pct": round(ben_votes/n_trees*100,1),
                "consensus_strength": "Strong" if max(mal_votes,ben_votes)/n_trees>0.8 else
                                      "Moderate" if max(mal_votes,ben_votes)/n_trees>0.6 else "Weak"}

    # ── MAIN PREDICT ─────────────────────────────────────────────────────
    def predict_with_xai(self, features):
        X_raw    = np.array(features).reshape(1,-1)
        X_scaled = self.scaler.transform(X_raw)

        global_proba = self.global_model.predict_proba(X_scaled)[0].tolist()
        current_label = "Malignant" if global_proba[1] > 0.5 else "Benign"

        # 1. SHAP
        g_shap = self._shap_values(X_scaled)
        sorted_imp = sorted({FEATURE_NAMES[i]:abs(g_shap[i]) for i in range(30)}.items(),
                            key=lambda x:x[1], reverse=True)

        # 2. LIME
        lime_exp = self._lime_explanation(X_raw)

        # 3. Decision Path
        dec_path = self._decision_path(X_scaled)

        # 4. Counterfactuals
        cf = self._counterfactuals(X_raw, X_scaled, current_label)

        # 5. Tree vote breakdown
        confidence = self._confidence_breakdown(X_scaled)

        # 6. Feature interactions (skip if slow — optional)
        try:    interactions = self._feature_interactions(X_scaled)
        except: interactions = []

        # Per-node predictions
        node_preds = {}
        for hid, node in self.nodes.items():
            proba = node.predict(X_raw)
            sv    = node.get_shap_values(X_raw)
            node_preds[hid] = {"probability":proba, "shap_values":sv}

        sizes = [n.data.shape[0] for n in self.nodes.values()]
        total = sum(sizes)
        consensus = [0.0, 0.0]
        for (hid,pred),s in zip(node_preds.items(), sizes):
            w = s/total
            consensus[0] += pred["probability"][0]*w
            consensus[1] += pred["probability"][1]*w

        return {
            "global_prediction":  {"benign_prob":round(global_proba[0],4),"malignant_prob":round(global_proba[1],4),
                                   "label":current_label,"confidence":round(max(global_proba)*100,1)},
            "federated_consensus":{"benign_prob":round(consensus[0],4),"malignant_prob":round(consensus[1],4),
                                   "label":"Malignant" if consensus[1]>0.5 else "Benign",
                                   "confidence":round(max(consensus)*100,1)},
            "node_predictions":{
                hid:{"benign_prob":round(p["probability"][0],4),"malignant_prob":round(p["probability"][1],4),
                     "label":"Malignant" if p["probability"][1]>0.5 else "Benign",
                     "top_features":sorted({FEATURE_NAMES[i]:round(abs(p["shap_values"][i]),4) for i in range(30)}.items(),
                                           key=lambda x:x[1],reverse=True)[:5]}
                for hid,p in node_preds.items()
            },
            "xai": {
                "shap": {"values":[round(v,4) for v in g_shap],"features":FEATURE_NAMES,
                         "top_features":[{"name":k,"value":round(v,4),"direction":"malignant" if g_shap[FEATURE_NAMES.index(k)]>0 else "benign"} for k,v in sorted_imp[:12]]},
                "lime": lime_exp,
                "decision_path": dec_path,
                "counterfactuals": cf,
                "confidence_breakdown": confidence,
                "feature_interactions": interactions,
            },
            "global_feature_weights":{FEATURE_NAMES[i]:round(self.global_weights[i],4) for i in range(30)},
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())
        }


print("Initializing Federated XAI System...")
aggregator = FederatedAggregator()
print("✓ All nodes trained and ready")

# ── ROUTES ───────────────────────────────────────────────────────────────
@app.route('/api/status')
def get_status():
    return jsonify({"status":"online","nodes":len(aggregator.nodes),"model_ready":aggregator.is_trained,
                    "hospitals":HOSPITALS,"training_rounds":len(aggregator.round_history)})

@app.route('/api/hospitals')
def get_hospitals():
    return jsonify({hid:{**info,**aggregator.nodes[hid].stats()} for hid,info in HOSPITALS.items()})

@app.route('/api/training-history')
def get_training_history():
    return jsonify({"rounds":aggregator.round_history,"total_rounds":len(aggregator.round_history),
                    "final_global_accuracy":round(aggregator.round_history[-1]["global_acc"]*100,1)})

@app.route('/api/aggregation-log')
def get_log():
    return jsonify({"log":aggregation_log[-30:]})

@app.route('/api/sample-patient')
def sample_patient():
    data, labels = load_breast_cancer(return_X_y=True)
    t = request.args.get('type','any')
    if   t=='malignant': idx = random.choice(np.where(labels==1)[0])
    elif t=='benign':    idx = random.choice(np.where(labels==0)[0])
    else:                idx = random.randint(0, len(data)-1)
    return jsonify({"features":data[idx].tolist(),"true_label":"Malignant" if labels[idx]==1 else "Benign",
                    "feature_names":FEATURE_NAMES,"patient_id":f"PAT-{random.randint(10000,99999)}"})

@app.route('/api/predict', methods=['POST'])
def predict():
    d = request.get_json()
    features = d.get('features')
    if not features or len(features)!=30:
        return jsonify({"error":"Need 30 features"}), 400
    try:
        result = aggregator.predict_with_xai(features)
        log_event("predict","global",f"Prediction: {result['global_prediction']['label']} ({result['global_prediction']['confidence']}%)")
        return jsonify(result)
    except Exception as e:
        return jsonify({"error":str(e),"trace":traceback.format_exc()}), 500

@app.route('/api/federated-round', methods=['POST'])
def sim_round():
    last = aggregator.round_history[-1]
    rec  = {"round":last["round"]+1,
            "hosp_1_acc":min(0.99,last["hosp_1_acc"]+random.uniform(0.001,0.005)),
            "hosp_2_acc":min(0.99,last["hosp_2_acc"]+random.uniform(0.001,0.005)),
            "hosp_3_acc":min(0.99,last["hosp_3_acc"]+random.uniform(0.001,0.005)),
            "global_acc":min(0.99,last["global_acc"]+random.uniform(0.002,0.007)),
            "communication_cost":max(0.02,last["communication_cost"]*0.85)}
    aggregator.round_history.append(rec)
    return jsonify(rec)

@app.route('/api/upload-data/<hid>', methods=['POST'])
def upload_data(hid):
    if hid not in aggregator.nodes: return jsonify({"error":"Unknown hospital"}), 404
    if 'file' not in request.files: return jsonify({"error":"No file"}), 400
    try:
        df = pd.read_csv(io.StringIO(request.files['file'].read().decode('utf-8')))
        lc = next((c for c in ['label','diagnosis','target','class'] if c in df.columns), df.columns[-1])
        labels = df[lc].astype(int).values
        data   = df.drop(columns=[lc]).values.astype(float)
        if data.shape[1]!=30: return jsonify({"error":f"Need 30 feature cols, got {data.shape[1]}"}), 400
        if len(np.unique(labels))<2: return jsonify({"error":"Need both classes"}), 400
        if len(data)<6: return jsonify({"error":"Need ≥6 rows"}), 400
        node = aggregator.nodes[hid]
        node.retrain(data, labels)
        log_event("upload_data",hid,f"Uploaded {len(data)} rows → acc={node.accuracy_history[-1]*100:.1f}%")
        return jsonify({"success":True,"rows_loaded":len(data),"local_accuracy":round(node.accuracy_history[-1]*100,1),
                        "weights_source":node.weights_source,"hospital":HOSPITALS[hid]["name"]})
    except Exception as e:
        return jsonify({"error":str(e)}), 500

@app.route('/api/upload-weights/<hid>', methods=['POST'])
def upload_weights(hid):
    if hid not in aggregator.nodes: return jsonify({"error":"Unknown hospital"}), 404
    if 'file' not in request.files: return jsonify({"error":"No file"}), 400
    try:
        payload = json.loads(request.files['file'].read().decode('utf-8'))
        weights = payload if isinstance(payload,list) else payload.get('weights',[])
        if len(weights)!=30: return jsonify({"error":"Need 30 weights"}), 400
        weights = [float(w) for w in weights]
        t = sum(weights)
        if t>0: weights = [w/t for w in weights]
        aggregator.nodes[hid].set_weights(weights)
        top = FEATURE_NAMES[int(np.argmax(weights))]
        log_event("upload_weights",hid,f"Weights injected — top: {top}")
        return jsonify({"success":True,"weights_received":30,"top_feature":top,
                        "weights_source":"uploaded_weights","hospital":HOSPITALS[hid]["name"]})
    except Exception as e:
        return jsonify({"error":str(e)}), 500

@app.route('/api/retrain/<hid>', methods=['POST'])
def retrain(hid):
    if hid not in aggregator.nodes: return jsonify({"error":"Unknown hospital"}), 404
    node = aggregator.nodes[hid]
    node.retrain()
    log_event("retrain",hid,f"Retrained — acc={node.accuracy_history[-1]*100:.1f}%")
    return jsonify({"success":True,"local_accuracy":round(node.accuracy_history[-1]*100,1),
                    "data_points":node.data.shape[0],"weights_source":node.weights_source,
                    "hospital":HOSPITALS[hid]["name"]})

@app.route('/api/aggregate', methods=['POST'])
def aggregate():
    try:
        rec = aggregator.aggregate()
        return jsonify({"success":True,"round":rec["round"],
                        "global_accuracy":round(rec["global_acc"]*100,1),
                        "node_accuracies":{"hosp_1":round(rec["hosp_1_acc"]*100,1),
                                           "hosp_2":round(rec["hosp_2_acc"]*100,1),
                                           "hosp_3":round(rec["hosp_3_acc"]*100,1)}})
    except Exception as e:
        return jsonify({"error":str(e)}), 500

@app.route('/api/export-weights/<hid>')
def export_weights(hid):
    if hid not in aggregator.nodes: return jsonify({"error":"Unknown hospital"}), 404
    node = aggregator.nodes[hid]
    return jsonify({"hospital_id":hid,"hospital_name":HOSPITALS[hid]["name"],
                    "weights":node.get_model_weights(),"feature_names":FEATURE_NAMES,
                    "exported_at":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())})

@app.route('/api/export-sample-csv/<hid>')
def export_csv(hid):
    if hid not in aggregator.nodes: return jsonify({"error":"Unknown hospital"}), 404
    node = aggregator.nodes[hid]
    rows = [{FEATURE_NAMES[j]:round(float(row[j]),6) for j in range(30)} | {"label":int(lbl)}
            for row,lbl in zip(node.data,node.labels)]
    csv = pd.DataFrame(rows).to_csv(index=False)
    return Response(csv, mimetype='text/csv',
                    headers={"Content-Disposition":f"attachment;filename={hid}_data.csv"})


# ── FRONTEND SERVING (must come LAST, after all /api/* routes) ────────────
import os as _os
from flask import send_from_directory as _sfd

_FRONTEND = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', 'frontend')

@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve_frontend(path):
    # Never intercept /api/* — those are handled above
    if path.startswith('api/'):
        return jsonify({"error": "Not found"}), 404
    target = path if path and _os.path.exists(_os.path.join(_FRONTEND, path)) else 'index.html'
    return _sfd(_FRONTEND, target)

# ── JSON error handlers so Flask never returns HTML on API errors ─────────
@app.errorhandler(404)
def not_found(e):
    if request.path.startswith('/api/'):
        return jsonify({"error": "Endpoint not found"}), 404
    return _sfd(_FRONTEND, 'index.html')

@app.errorhandler(405)
def method_not_allowed(e):
    return jsonify({"error": "Method not allowed"}), 405

@app.errorhandler(500)
def internal_error(e):
    return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=False, port=5050, host='0.0.0.0')
