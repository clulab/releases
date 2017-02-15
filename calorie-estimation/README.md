# `questions.csv`
This comma-separated spreadsheet summarizes the food items that were used in our calorie estimation quiz. The columns are:

- `label`: The name identifying the food.
- `ingredients`: *single* for single-ingredient foods and *mixed* for mixed-ingredient foods.
- `calGold`: The ground-truth number of (kilo)calories in the food item.
- `gGold`: The ground-truth weight in grams of the food item.
- `isScaled`: *scaled* means there is some standard-sized reference object for scaling, and *unscaled* means there isn't.


# `responses.csv`

This comma-separated spreadsheet lists all estimates and demographic information provided by each respondent, one row per respondent. The columns are:

- `id`: An integer identification number assigned at random to each respondent. Notice that the IDs for the expert respondents begin again at 1.
- `q1`–`q20`: Predictions in calories for each question (corresponding to the rows in `questions.csv`.
- `conf`: Confidence in the estimations, as reported by the respondent, ranging from 0 to 100. 50 is the default response.
- `age`: Age range as reported by the respondent, in years.
- `gender`: Gender as reported by the respondent. Only *Female* and *Male* responses were used in analysis.
- `bmi`: Body Mass Index (BMI, kg/m^2), as reported by the respondents.
- `expert`: *expert* means the respondent was selected by the authors as an expert nutritionist, and *nonexpert* means they were not. No effort was made to ensure that nutrition experts did not take the quiz without the authors' direct knowledge.
