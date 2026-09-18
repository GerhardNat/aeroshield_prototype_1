"""
AeroShield - 08: Text mining / NLP on incident-report text (C8, session 9 evidence)
Preprocessing, regex-based entity/indicator extraction, and a TF-IDF + Logistic
Regression classifier for incident category, with evaluation.
"""
import os
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import numpy as np
import re
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

NAVY = "#1B2A4A"; BLUE = "#2E5AAC"; TEAL = "#1F7A6C"

df = pd.read_csv("data/incident_reports.csv")

# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
def clean_text(t):
    t = t.lower()
    t = re.sub(r"[^a-z0-9\s\-]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

df["clean_text"] = df.description.apply(clean_text)

# ---------------------------------------------------------------------------
# Entity / indicator extraction (regex-based, appropriate for structured synthetic tickets)
# ---------------------------------------------------------------------------
USER_RE = re.compile(r"\b(STF\d{4}|VND\d{4})\b")
DEVICE_RE = re.compile(r"\bDEV\d{4}\b")
APP_RE = re.compile(r"\b(Baggage-Handling-System|Gate-Management-System|Departure-Control-System|"
                     r"Airfield-Ops-Portal|Cargo-Manifest-System)\b")
VENDOR_RE = re.compile(r"\b(SkyLink Ground Services|AeroFuel Solutions|Coastal Catering Co|"
                        r"Falcon Cargo Handling|Meridian Aircraft Maintenance|Horizon Cleaning Services)\b")

def extract_entities(text):
    return {
        "accounts": USER_RE.findall(text),
        "devices": DEVICE_RE.findall(text),
        "applications": APP_RE.findall(text),
        "vendors": VENDOR_RE.findall(text),
    }

df["entities"] = df.description.apply(extract_entities)
n_with_entity = df.entities.apply(lambda e: any(len(v) for v in e.values())).sum()
print(f"Tickets with at least one extracted entity/indicator: {n_with_entity}/{len(df)}")

entity_records = []
for _, r in df.iterrows():
    for etype, vals in r.entities.items():
        for v in set(vals):
            entity_records.append({"ticket_id": r.ticket_id, "entity_type": etype, "value": v})
entities_df = pd.DataFrame(entity_records)
entities_df.to_csv("outputs/extracted_entities.csv", index=False)

# ---------------------------------------------------------------------------
# Classification: predict category from text (TF-IDF + Logistic Regression)
# ---------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    df.clean_text, df.category, test_size=0.25, random_state=821, stratify=df.category
)
tfidf = TfidfVectorizer(max_features=800, ngram_range=(1, 2), min_df=2)
Xtr = tfidf.fit_transform(X_train)
Xte = tfidf.transform(X_test)

clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=821)
clf.fit(Xtr, y_train)
y_pred = clf.predict(Xte)

acc = accuracy_score(y_test, y_pred)
f1_macro = f1_score(y_test, y_pred, average="macro")
report = classification_report(y_test, y_pred, output_dict=True)
print(f"\nText classification accuracy={acc:.3f}  macro-F1={f1_macro:.3f}")

# ---------------------------------------------------------------------------
# Topic modelling (LDA) as a complementary unsupervised view
# ---------------------------------------------------------------------------
cv = CountVectorizer(max_features=500, stop_words="english", min_df=3)
dtm = cv.fit_transform(df.clean_text)
lda = LatentDirichletAllocation(n_components=6, random_state=821, max_iter=25)
lda.fit(dtm)
terms = cv.get_feature_names_out()
topics = []
for idx, comp in enumerate(lda.components_):
    top_terms = [terms[i] for i in comp.argsort()[-8:][::-1]]
    topics.append({"topic": idx, "top_terms": top_terms})
    print(f"Topic {idx}: {', '.join(top_terms)}")

# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
cats = sorted(df.category.unique())
cm_df = pd.crosstab(y_test, y_pred).reindex(index=cats, columns=cats, fill_value=0)
im = axes[0].imshow(cm_df.values, cmap="Blues")
axes[0].set_xticks(range(len(cats))); axes[0].set_xticklabels(cats, rotation=60, ha="right", fontsize=6.5)
axes[0].set_yticks(range(len(cats))); axes[0].set_yticklabels(cats, fontsize=6.5)
axes[0].set_title("Incident-category classification\n(confusion matrix)", fontsize=9.5, color=NAVY, weight="bold")

entity_counts = entities_df.entity_type.value_counts()
axes[1].bar(entity_counts.index, entity_counts.values, color=TEAL)
axes[1].set_title("Extracted entities/indicators by type", fontsize=9.5, color=NAVY, weight="bold")
plt.tight_layout()
plt.savefig("outputs/figures/fig10_nlp_evaluation.png", dpi=180, facecolor="white")
plt.close()

metrics = {
    "classification": {"accuracy": round(acc, 4), "macro_f1": round(f1_macro, 4), "report": report},
    "topics": topics,
    "entity_extraction": {"tickets_with_entity": int(n_with_entity), "total_tickets": len(df),
                           "entity_counts_by_type": entity_counts.to_dict()},
}
with open("outputs/nlp_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2, default=str)

print("Saved NLP outputs.")
