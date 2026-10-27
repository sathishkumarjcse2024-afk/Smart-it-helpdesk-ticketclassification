import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import clean_text, infer_priority

def test_masks_email_and_secret():
    cleaned = clean_text("Contact me at test@example.com password: abc12345")
    assert "[EMAIL]" in cleaned
    assert "[REDACTED]" in cleaned

def test_priority_rules():
    assert infer_priority("critical network outage") == "High"
    assert infer_priority("application is slow") == "Medium"
    assert infer_priority("please install text editor") == "Low"
