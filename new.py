from flask import Flask, render_template
import pandas as pd
from sklearn.cluster import KMeans


covid_data1 = pd.read_csv(r"C:\Users\Saad Rehman\Downloads\project\covid_19_data.csv")
covid_list_data1 = pd.read_csv(r"C:\Users\Saad Rehman\Downloads\project\COVID19_line_list_data_modified.csv")

covid_list_data1["age"] = covid_list_data1["age"].fillna(covid_list_data1["age"].mean())

kmeans_model_1 = KMeans(n_clusters=3)
covid_data1["cluster"] = kmeans_model_1.fit_predict(
    covid_data1[["Confirmed", "Deaths", "Recovered"]]
)

kmeans_model_2 = KMeans(n_clusters=3)
covid_list_data1["cluster"] = kmeans_model_2.fit_predict(covid_list_data1[["age"]])

highest_affected = (
    covid_data1.groupby("Country/Region")["Confirmed"]
    .sum()
    .sort_values(ascending=False)
    .head(1)
)

second_highest_affected = (
    covid_data1.groupby("Country/Region")["Confirmed"]
    .sum()
    .sort_values(ascending=False)
    .iloc[1]
    .astype(str)
)


mortality_recovery_ratio = covid_data1["Deaths"].sum() / covid_data1["Recovered"].sum()
age_gender_tendencies = covid_list_data1.groupby(["gender", "age"]).size()
covid_list_data1["death"] = pd.to_datetime(covid_list_data1["death"], errors="coerce")
mortality_rate_age_groups = covid_list_data1.groupby("age")["death"].mean()


app = Flask(__name__)


@app.route("/")
def index():
    return render_template(
        "index.html",
        highest_affected=highest_affected,
        second_highest_affected=second_highest_affected,
        mortality_recovery_ratio=mortality_recovery_ratio,
        age_gender_tendencies=age_gender_tendencies,
        mortality_rate_age_groups=mortality_rate_age_groups,
        kmeans_clusters_1=covid_data1,
        kmeans_clusters_2=covid_list_data1,
    )


if __name__ == "__main__":
    app.run(debug=True)
