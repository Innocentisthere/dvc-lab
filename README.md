Python 3.11
```bash
git clone https://github.com/Innocentisthere/dvc-lab
cd dvc-lab
python -m venv myenv
pip install -r requirements.txt
dvc get https://github.com/Innocentisthere/jenkins-lab insurance.csv -o  data/insurance.csv
mlflow server --host 127.0.0.1 --port 8080
dvc repro
```
