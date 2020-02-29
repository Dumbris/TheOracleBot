install:
	pip install -r requirements.txt
	python src/download_bert.py
	python gpt/download_model.py 1558M
run:
	python src/start_bot.py
run2:
	python src/bot2.py
