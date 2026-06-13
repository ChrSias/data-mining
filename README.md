# Εξόρυξη Δεδομένων και Αλγόριθμοι Μάθησης

Εργαστηριακή Άσκηση — Ανάλυση Δικτυακής Κίνησης και Ανίχνευση Κυβερνοεπιθέσεων με το dataset **CIC-IDS-2017**.

**Τμήμα Μηχανικών Η/Υ & Πληροφορικής, Πανεπιστήμιο Πατρών** — Εαρινό Εξάμηνο 2025-2026

## Ομάδα

- Σωτήριος Κασσίδης 
- Χρήστος Σιάσιος

## Δομή του project

```
.
├── question1_eda (1).py            # Ερώτημα 1: Διερευνητική Ανάλυση Δεδομένων
├── question2_classification (1).py # Ερώτημα 2: Κατηγοριοποίηση (LR, DT, RF)
├── question3_clustering (1).py     # Ερώτημα 3: Ομαδοποίηση (K-Means, Hierarchical, DBSCAN)
├── requirements.txt                # Απαιτούμενες βιβλιοθήκες
├── report.pdf                      # Πλήρης αναφορά
├── report.docx                     # Αναφορά (Word source)
└── output/                         # Γραφήματα και αρχεία αποτελεσμάτων
```

## Εγκατάσταση

```bash
pip install -r requirements.txt
```

Απαιτούμενες βιβλιοθήκες: NumPy, Pandas, Matplotlib, Seaborn, scikit-learn, SciPy

## Dataset

Το dataset CIC-IDS-2017 ΔΕΝ συμπεριλαμβάνεται στο repo λόγω μεγέθους (~844 MB). Μπορεί να ληφθεί από:

- [Kaggle: Network Intrusion Dataset](https://www.kaggle.com/datasets/chethuhn/network-intrusion-dataset/)
- [Επίσημη σελίδα CIC](https://www.unb.ca/cic/datasets/ids-2017.html)

Τοποθέτησε τα 8 CSV αρχεία στον φάκελο `data/` στη ρίζα του project.

## Εκτέλεση

Σειρά υποχρεωτική (το Q2 και Q3 χρειάζονται το `cleaned_dataset.csv` που παράγει το Q1):

```bash
python "question1_eda (1).py"           # Παράγει cleaned_dataset.csv
python "question2_classification (1).py" # Ταξινόμηση
python "question3_clustering (1).py"     # Ομαδοποίηση
```

## Σύντομη Σύνοψη Αποτελεσμάτων

### Ερώτημα 1 — EDA
- 2.830.743 αρχικές γραμμές → **2.515.573 γραμμές × 45 features** μετά τον καθαρισμό
- Έντονη ανισορροπία: BENIGN ~83%, σπάνιες κλάσεις (Heartbleed: 11 δείγματα)

### Ερώτημα 2 — Κατηγοριοποίηση
| Μοντέλο | Accuracy | Weighted F1 | Macro F1 |
|---|---|---|---|
| Logistic Regression | 0.917 | 0.940 | 0.660 |
| Decision Tree | 0.998 | 0.998 | 0.957 |
| **Random Forest** | **0.998** | **0.998** | **0.961** |

### Ερώτημα 3 — Ομαδοποίηση
| Αλγόριθμος | Silhouette | Davies-Bouldin | ARI | NMI |
|---|---|---|---|---|
| K-Means (k=2) | **0.454** | 1.858 | **0.299** | 0.179 |
| Hierarchical (ward) | 0.427 | 1.932 | 0.213 | 0.146 |
| DBSCAN (eps=2.13) | 0.395 | **0.749** | 0.088 | **0.238** |
