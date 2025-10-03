install:
	pip install --upgrade pip
	pip install -r requirements.txt

train:
	python train.py

eval:
	echo "## Model Metrics" > report.md
	cat Results/metrics.txt >> report.md
	echo '\n## Confusion Matrix Plot' >> report.md
	echo '![Confusion Matrix](./Results/model_results.png)' >> report.md
	cml comment create report.md

update-branch:
	git config --global user.name "varshavenkatesan"
	git config --global user.email "varshavenkatesan@gmail.com"
	git add Model Results
	git commit -m "Update model and results" || echo "No changes to commit"
	git push --force origin HEAD:update

hf-login:
	git pull origin update || true
	git switch update || git checkout -b update
	pip install -U "huggingface_hub[cli]"
	huggingface-cli login --token $(HF) --add-to-git-credential

push-hub:
	huggingface-cli upload varshavenkatesan/MachineLearning ./App --repo-type=space --commit-message="Sync App files"
	huggingface-cli upload varshavenkatesan/MachineLearning ./Model --repo-type=space --commit-message="Sync Model"
	huggingface-cli upload varshavenkatesan/MachineLearning ./Results --repo-type=space --commit-message="Sync Results"

deploy: hf-login push-hub