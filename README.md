# Summative Lab: Classification, Recommendation and Clustering with PCA

**Author:** Carole Wambui
**Notebook:** `summative_lab.ipynb`
**Tools:** Python, Jupyter, pandas, numpy, scikit-learn, matplotlib, seaborn

As a junior data scientist at DataVine Analytics, I built three prototype machine learning solutions for client projects, following the CRISP-DM process.

## 1. Business Understanding
- **Wine distributor:** classify wines by their chemical properties for inventory and quality control (k-NN).
- **Agricultural supplier:** recommend feeds that give similar chick growth (cosine-similarity recommender).
- **Crime analysis:** group US states with similar crime patterns (K-Means and GMM clustering).

## 2. Data Understanding
- **Wine:** 178 wines, 13 chemical features, 3 cultivars.
- **Chickwts:** 71 chicks, a weight (g) and one of 6 feed types.
- **USArrests:** 50 states with Murder, Assault and Rape rates (per 100,000 residents) and UrbanPop (%).

## 3. Data Preparation
- There were no missing values in any dataset.
- One duplicate row in Chickwts was kept, because it is two real chicks with the same weight.
- Cleaned the text labels, checked value ranges and set state names as the USArrests index.
- Standardised all numeric features with StandardScaler, which distance-based methods need.

## 4. Modeling
- **k-NN (Wine):** converted the labels to numbers, then built a pipeline of scaling, PCA keeping 95% of the variance (10 components) and k-NN. GridSearchCV picked k = 10 with Euclidean distance.
- **Recommender (Chickwts):** standardised weight, applied PCA to 1 component, built a performance profile for each feed and computed cosine similarity between the feeds.
- **Clustering (USArrests):** chose the top 3 features (Assault, Rape, Murder) and reduced them to 2 PCA components. The elbow method for K-Means and BIC for GMM both chose 2 clusters.

## 5. Evaluation
- **Wine:** about 96.5% cross-validated accuracy, and 100% on the test set.
- **Feeds:** sunflower and casein are the most similar feeds (0.97), followed by linseed and soybean (0.95). Horsebean gives the lowest weights.
- **Crime:** a higher-crime group of about 20 states and a lower-crime group of about 30. K-Means and GMM agree on 49 of the 50 states, and GMM flags borderline states such as Missouri.
- **Limitations:** the datasets are small, and the crime data is historical.

## 6. Deployment
- Use the wine model as an automated quality-control check.
- Offer feed recommendations as a "similar products" feature once cost data is added.
- Use the crime clusters to prioritise resources, and review the borderline states individually.

