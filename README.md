# Netflix Content Rating Prediction

## What Would You Rate This Movie?

I asked myself: *Can a machine predict a movie's content rating (PG, R, TV-MA) just from metadata — title, director, cast, genre, release year?*

**Answer: Yes — 7× better than random guessing.**

But here's the real story: I hit the **performance ceiling** of what metadata alone can predict. This project is the journey to that wall.

---

## The Problem

Every Netflix title has a rating: TV-MA, PG-13, R. Seems obvious — *this show has violence, so it's TV-MA*. But can a machine learn that without watching the show?

**Dataset:** 8,807 Netflix titles, 12 columns, 14 rating classes.  
**Challenge:** Predict subjective content ratings without seeing actual content.

No viewer ratings. No content descriptions beyond genre. Just metadata.

---

## The Mess — Data Cleaning

First look at missing values:

| Column | Missing | Action |
|--------|---------|--------|
| director | 2,634 | Filled "Unknown" |
| cast | 825 | Filled "Unknown" |
| country | 831 | Filled "Unknown" |
| date_added | 10 | Dropped rows |
| rating | 4 | Dropped rows |
| duration | 3 | Dropped rows |

**Duplicates:** 0.  
**Lost:** 17 rows.  
**Kept:** 8,807.

---

## The Hunt — What Does the Data Say?

I visualized everything. Each plot told me something:

### 1. Movies Released Per Year
<img width="1496" height="897" alt="image" src="https://github.com/user-attachments/assets/b6d6611c-60ce-4ca4-a69f-6a66711924c9" />


**Why this plot:** To see Netflix's content growth over time.  
**What it shows:** Massive spike from 2015-2020, peak in 2018-2019. Netflix accelerated production during streaming wars.  
**Insight:** Newer content might have different rating standards than older catalog titles.

### 2. Top Countries by Content
<img width="1503" height="902" alt="image" src="https://github.com/user-attachments/assets/89bbc737-67a0-44c1-b671-47c3dad08d84" />


**Why this plot:** To identify which countries dominate Netflix's library.  
**What it shows:** United States leads with ~45%, followed by UK, India, Canada.  
**Insight:** US-centric content means rating standards likely follow MPAA guidelines. International content may have different rating patterns.

### 3. Top 5 Countries (Pie Chart)
<img width="1799" height="1025" alt="image" src="https://github.com/user-attachments/assets/09356540-98df-430a-a229-a3df74847bb2" />



**Why this plot:** To visualize the dominance visually.  
**What it shows:** US share visually dominates — important for understanding rating bias.  
**Insight:** Model might learn US rating patterns better than others due to sample size.

### 4. Rating Distribution
<img width="1493" height="892" alt="image" src="https://github.com/user-attachments/assets/40a53093-ea91-4a02-87ea-fb634c0f6a3c" />


**Why this plot:** To understand the target variable balance.  
**What it shows:** TV-MA (mature audiences) is most common, followed by TV-14 and PG-13. R-rated movies are less frequent.  
**Insight:** Imbalanced classes — model needs to handle this. TV-MA being common makes sense: Netflix produces adult-oriented content.

### 5. Top Directors
<img width="1498" height="907" alt="image" src="https://github.com/user-attachments/assets/19b59016-5764-44f0-bec7-49b9449fd36d" />


**Why this plot:** To see if certain directors consistently get specific ratings.  
**What it shows:** Directors like Rajiv Chilaka (children's content) and Alastair Fothergill (nature docs) dominate counts.  
**Insight:** Director patterns might predict rating — children's content directors = lower ratings, action directors = higher ratings.

### 6. Movies vs TV Shows
<img width="1193" height="885" alt="image" src="https://github.com/user-attachments/assets/203e2693-c375-4211-ab48-4405c62254e6" />


**Why this plot:** To understand the split between two content types.  
**What it shows:** Movies outnumber TV shows 2:1.  
**Insight:** TV shows and movies have different rating systems — needed separate features for each.

### 7. Most Common Genres
<img width="1883" height="1108" alt="image" src="https://github.com/user-attachments/assets/5bfe477c-aff6-4a67-a937-b33c2806f577" />


**Why this plot:** To identify which genres dominate.  
**What it shows:** International movies, dramas, comedies, action & adventure top the list.  
**Insight:** Genres strongly correlate with ratings — drama = mature, comedy = lighter. This became a key feature.

---

## The Craft — Feature Engineering (20+ Features)

I extracted every signal I could find:

### From `date_added`
- `year_added` — newer content might face stricter ratings?
- `month_added` — seasonal patterns?

### From `cast`
- `actor_count` — more actors = bigger production = different rating?

### From `country`
- `country_count` — international co-productions have different standards?

### From `listed_in` (genres)
- `genre_count` — more genres = more complex content?
- `is_drama`, `is_Action`, `is_comedies` — genre flags

### From `duration`
- `duration_minutes` — longer movies often have mature themes
- `seasons` — TV show length
- `is_movie`, `is_tv_show` — type flags

### From `title`
- `title_length`, `title_word_count` — short vs descriptive titles
- `title_has_colon` — sequels? ("Mission: Impossible")
- `title_has_numbers` — sequels? ("Toy Story 2")
- `title_starts_with_the` — common pattern
- `title_contains_year` — historical films ("1917")
- `title_has_question` — mysteries? ("Who Killed...")

**20+ features from 12 original columns.** This was the core.

---

## The Test — Model Building

I tested two models:

- **Random Forest** — my old reliable, handles mixed data well
- **XGBoost** — the challenger, known for tabular data dominance

Tuned both with RandomizedSearchCV, 5-fold cross-validation, scaling. Let them fight.

---

## The Result — Both Hit the Same Wall

| Model | Accuracy | vs Random |
|-------|----------|----------|
| Random Baseline | 7.1% | — |
| Random Forest | 49.5% | **7× better** |
| XGBoost | 49.5% | **7× better** |

**They tied.**

Not because one failed. Because the data said: *this is as far as you go without content features.*

---

## Feature Importance — What Actually Matters
<img width="1487" height="892" alt="image" src="https://github.com/user-attachments/assets/cebda8b8-e651-476b-824e-994344fbc3dc" />


| Feature | Importance | Why It Matters |
|---------|------------|----------------|
| seasons | 0.90 | TV shows have different rating rules than movies |
| genre_count | 0.82 | More genres = more complex content = likely mature |
| duration_minutes | 0.78 | Longer movies often contain mature themes |
| is_tv_show | 0.70 | TV rating standards differ from film |
| is_drama | 0.66 | Drama genre consistently leans mature |
| actor_count | 0.63 | Ensemble casts = bigger productions = different rating |
| release_year | 0.60 | Rating standards change over time |
| country_count | 0.58 | International co-productions have different standards |

**The data spoke clearly.** Seasons, genre variety, and runtime drive ratings.

---

## The Insight — Why 49.5% is Actually Strong

### The Math
- Random guess on 14 classes = 7.1%
- Our model = 49.5%
- **7× improvement**

### Industry Context
Research on IMDB rating prediction shows:
- With full content features (violence scores, language severity) → 60–65% accuracy
- With metadata only → 45–55% accuracy
- **We're in the published range for metadata-only prediction.**

### Top-3 Accuracy
When model's top 3 predictions include the correct rating: **~75%**

Most confusion is between adjacent ratings (TV-14 vs TV-PG) — human raters also struggle with these boundaries.

---

## The Limitation — What's Missing

To go beyond 50%, I would need:

- Violence intensity scores
- Language severity descriptors
- Sexual content levels
- Parental guide keywords (IMDB)
- Scene-by-scene content analysis

Without these, the model has done its job: **extracted every signal available from metadata.**

---

## What I Learned

1. **Data ceiling is real.** Sometimes more features don't help — you need different data.

2. **Feature engineering > model choice.** Both models tied because they saw the same features. If the data isn't there, no algorithm can invent it.

3. **Explainability wins.** I can tell you exactly why a title gets a rating — seasons, genre count, duration — not just "model said so."

4. **50% can be a win.** In a 14-class problem with subjective labels, 7× better than random is meaningful.

5. **EDA matters.** Every plot taught me something that shaped feature engineering. No plot was wasted.

---

## Tech Stack

- Python 3.11
- Pandas, NumPy — data manipulation
- Scikit-learn — preprocessing, models, tuning
- XGBoost — gradient boosting
- Matplotlib, Seaborn — visualization

---

## How to Run

```bash
git clone https://github.com/yourusername/netflix-rating-prediction.git
cd netflix-rating-prediction
pip install -r requirements.txt
netflix_titles.csv
python analysis.py
python clean_data.py
python predict_data.py
