.PHONY: install train test serve

install:
	pip install -r requirements.txt

train:
	python scripts/train.py

test:
	pytest tests/ -v

serve:
	uvicorn src.serving.api:app --host 0.0.0.0 --port 8000 --reload
