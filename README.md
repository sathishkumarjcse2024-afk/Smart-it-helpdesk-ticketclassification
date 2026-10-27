# Smart IT Helpdesk Ticket Classification

A student capstone implementation based on the supplied project brief. It classifies IT tickets, estimates priority, recommends a support queue, supports staff corrections, records override history, evaluates model metrics, and exports results.

## Features
- Single ticket form and CSV batch classification
- Text cleaning and masking of common email/secret patterns
- Categories: Account Access, Hardware, Software, Network, Application Support
- TF-IDF + Logistic Regression baseline classifier
- Confidence display and low-confidence review flag
- Rule-based priority estimate (High/Medium/Low)
- Queue recommendation with a simple evidence explanation
- Staff correction and reason saved to SQLite history
- Precision, recall, F1, confusion matrix, category volume chart
- CSV export for classified tickets and review history
- Synthetic sample data only; do not upload sensitive production tickets

## Run locally
1. Install Python 3.10 or later.
2. Open a terminal in this folder.
3. Create a virtual environment (recommended):
   - Windows: `python -m venv .venv` then `.venv\\Scripts\\activate`
   - macOS/Linux: `python3 -m venv .venv` then `source .venv/bin/activate`
4. Install packages: `pip install -r requirements.txt`
5. Start the app: `streamlit run app.py`
6. Open the local address shown in the terminal (usually `http://localhost:8501`).

## Sample CSV
Use `data/sample_tickets.csv`. For uploads, the only required column is `issue`; `ticket_id`, `service`, and `date` are optional.

## Project structure
- `app.py` — interface, preprocessing, model, routing, feedback, dashboard
- `data/sample_tickets.csv` — synthetic labelled examples
- `tests/test_preprocessing.py` — basic preprocessing tests
- `requirements.txt` — dependencies
- `helpdesk.db` — created automatically when feedback is saved

## Limitations and responsible use
- Sample data is synthetic and small, so model scores are only demonstrations.
- Confidence values are model scores, not guaranteed probabilities of correctness.
- Priority rules are simple keyword rules and should be reviewed by staff.
- The queue mapping is an editable example and must be checked against local service ownership.
- Low-confidence or unclear tickets require human review. Tickets are never closed automatically.
- Secret masking is a basic safety measure, not a guarantee; review uploaded data before use.
- Do not put credentials in source code. Use approved secrets management if adding external services.
