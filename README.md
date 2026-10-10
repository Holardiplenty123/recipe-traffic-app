# Tasty Bytes Homepage Recipe Picker

A machine learning web app that predicts which recipes will bring high traffic to a recipe site when they are featured on the homepage.

**Live app:** [https://tasty-bytes-recipe-picker.streamlit.app/]

Built on the Tasty Bytes recipe traffic case from the DataCamp Data Science Certification. The case and the data belong to DataCamp. This repository is my own work on top of it: the analysis, the models and the app.

## The problem

Tasty Bytes started in 2020, during Covid, as a recipe search engine that helped people cook with the limited supplies they had at home. It is now a subscription business with meal plans and a premium plan.

Every day one recipe is featured on the homepage, and it is chosen by hand. When the pick is popular, traffic to the rest of the site rises by as much as 40%, and more traffic means more subscriptions. The product team did not have a way to choose a popular recipe on purpose. Their goal was to correctly predict high traffic recipes 80% of the time.

## What the app does

- **Check a recipe.** Enter a recipe's details and get its chance of driving high traffic, plus how the same recipe would score in each category.
- **Pick today's recipe.** Add several candidates, or upload a CSV, and the app ranks them and names the one to feature.
- **What the data shows.** Traffic by category and how nutrition relates to the outcome.
- **How reliable is it.** Both models side by side, the precision and recall trade-off, and what pushes a prediction up or down.
- **Track it and next steps.** The number to monitor, its starting value, and a small calculator for tracking it weekly.

A slider in the sidebar sets how sure the model must be before it calls a recipe popular. A higher setting flags fewer recipes, but more of the flagged ones turn out to be right.

## Data and cleaning

The data has 947 recipes, with calories, carbohydrate, sugar, protein, category, servings and whether traffic was high.

- `recipe` is used as the unique identifier.
- `category` has "Chicken Breast" merged into "Chicken", leaving the ten categories in the data description.
- `servings` holds text such as "4 as a snack" in some rows, so the number is extracted and stored as an integer.
- `calories`, `carbohydrate`, `sugar` and `protein` have missing values, which are filled with the median of the recipe's own category.
- `high_traffic` is "High" or blank in the raw data. It is converted to 1 for High and 0 for blank.

## Models

This is a binary classification problem: will a recipe bring high traffic, yes or no?

- **Baseline:** Logistic Regression
- **Comparison:** Random Forest

Both use scaled numeric features and one-hot encoded categories, and were evaluated on a held-out 20% test set and with 5-fold cross-validation.

| Model | Test accuracy | Test precision |
|---|---|---|
| Logistic Regression | 0.78 | 0.85 |
| Random Forest | 0.76 | 0.80 |

Logistic Regression is the model in the app. It scored higher and is easier to explain to the product team. Cross-validated precision is about 0.80.

The metric I recommend the business monitors is **precision of homepage picks**: of the recipes featured, what share bring high traffic? About 61% of recipes in the data were high traffic, so that is the starting point with no model, and the goal is 80% or more.

## Limits

- The result sits close to the 80% goal, not far above it. The app is a decision aid, not a guarantee.
- 947 recipes is a small dataset.
- Missing nutrition values were filled before the train and test split, which can leak a little information. A pipeline that does the filling inside cross-validation would be cleaner.
- The data is missing things that are probably useful, such as time to make, cost per serving, ingredients and the day the recipe was featured.

## Recommended next steps

1. Run a four to eight week pilot, alternating model picks and manual picks, and compare hit rates.
2. Log every featured recipe and its outcome, to grow the dataset.
3. Add the missing recipe details as features.
4. Retrain monthly and review the model if the weekly hit rate drops below 80%.

## Run it yourself

```
pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

`train_model.py` cleans the data, trains both models, and saves the model and the numbers the app shows. The raw data file is `data/recipe_site_traffic_2212.csv`.

## Project files

```
app.py                 the Streamlit app
train_model.py         cleaning, training and evaluation
requirements.txt       package versions
.streamlit/config.toml colour theme
data/                  raw and cleaned data
model/                 saved model and metrics
```


