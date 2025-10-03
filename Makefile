install:
	pip install --upgrade pip
	pip install -r requirements.txt

train:
	python train.py

eval:
	echo "## Model Metrics" > report.md
	cat Results/metrics.txt >> report.md || echo "Metrics file not found"
	echo "" >> report.md
	echo "## Confusion Matrix Plot" >> report.md
	echo "![Confusion Matrix](./Results/model_results.png)" >> report.md

update-branch:
	git config --global user.name "varshavenkatesan"
	git config --global user.email "varshavenkatesan26@gmail.com"
	git add Model Results || echo "No Model/Results to add"
	git commit -m "Update model and results" || echo "No changes to commit"
	git push --force origin HEAD:update || echo "Push failed"

hf-login:
	git pull origin update || true
	git switch update || git checkout -b update
	pip install -U "huggingface_hub[cli]"
	huggingface-cli login --token $$HF --add-to-git-credential

push-hub:
	huggingface-cli upload varshavenkatesan/MachineLearning ./App --repo-type=space --commit-message="Sync App files" || echo "App upload failed"
	huggingface-cli upload varshavenkatesan/MachineLearning ./Model --repo-type=space --commit-message="Sync Model" || echo "Model upload failed"
	huggingface-cli upload varshavenkatesan/MachineLearning ./Results --repo-type=space --commit-message="Sync Results" || echo "Results upload failed"

deploy: hf-login push-hub