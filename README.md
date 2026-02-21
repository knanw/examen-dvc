# Examen DVC et Dagshub
Dans ce dépôt vous trouverez l'architecture proposé pour mettre en place la solution de l'examen. 

```bash       
├── examen_dvc          
│   ├── data       
│   │   ├── processed      
│   │   └── raw       
│   ├── metrics       
│   ├── models      
│   │   ├── data      
│   │   └── models        
│   ├── src       
│   └── README.md.py       
```
N'hésitez pas à rajouter les dossiers ou les fichiers qui vous semblent pertinents.

Vous devez dans un premier temps *Fork* le repo et puis le cloner pour travailler dessus. Le rendu de cet examen sera le lien vers votre dépôt sur DagsHub. Faites attention à bien mettre https://dagshub.com/licence.pedago en tant que colaborateur avec des droits de lecture seulement pour que ce soit corrigé.

Vous pouvez télécharger les données à travers le lien suivant : https://datascientest-mlops.s3.eu-west-1.amazonaws.com/mlops_dvc_fr/raw.csv.

###
1. clone project
2. dvc fetch data/raw.dvc (.dvc file)
3. dvc checkout 
4. dvc pull (combines dvc fetch and dvc checkout!)

### workflow:
1. python code ausführen python ./src/models/train_model.py
2. dvc add models/trained_model.joblib (dvc add ausführen)
3. git add metrics/accuracy.json models/trained_model.joblib.dvc (git add für kleine files wie erzeugte *.dvc files)
4. git commit -m "Trained and evaluated a first Random Forest Classifier"
5. dvc push
6. git push 
...



### Pipeline set up
1. 
dvc stage add -n download_data \
              -d src/data/import_raw_data.py \
              -o data/raw_data/raw.csv \
              python src/data/import_raw_data.py

2. dvc repro

3. 

dvc stage add -n prepare \
              -d src/data/models/make_dataset.py \
              -d data/raw_data/raw.csv \
              -o data/processed_data \
              python src/data/models/make_dataset.py


dvc stage add -n train \
              -d src/data/models/train_model.py \
              -d data/processed_data \
              -o models/scaler.pkl \
              -o models/best_params.pkl \
              -o models/trained_model.joblib \
              python src/models/train_model.py

dvc stage add -n evaluation \
              -d src/models/evaluate_model.py 
              -d models/scaler.pkl \
              -d models/trained_model.joblib \
              -M metrics/scores.json \
              python src/models/evaluate_model.py
dvc repro