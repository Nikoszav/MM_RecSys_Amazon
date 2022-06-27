import config
import pandas as pd

if __name__ == "__main__":
    rating_dataset = pd.read_pickle(config.ratings_file)
    user_df = pd.read_csv(
        "D:/University/Edinburgh/Dissertation/data/movielens/users.dat",
        sep="::",
        header=None,
    )
    user_df.rename(
        columns={0: "userId", 1: "gender", 2: "age", 3: "occupation", 4: "postcode"},
        inplace=True,
    )
    # user_df.drop("postcode")
    user_df.loc[user_df["age"] == 1, "Age Group"] = "Under 18"
    user_df.loc[user_df["age"] == 18, "Age Group"] = "18-24"
    user_df.loc[user_df["age"] == 25, "Age Group"] = "25-34"
    user_df.loc[user_df["age"] == 35, "Age Group"] = "35-44"
    user_df.loc[user_df["age"] == 45, "Age Group"] = "45-49"
    user_df.loc[user_df["age"] == 50, "Age Group"] = "50-55"
    user_df.loc[user_df["age"] == 56, "Age Group"] = "56+"

    dummies_gender = pd.get_dummies(user_df[["gender"]])
    dummies_age = pd.get_dummies(user_df[["Age Group"]])
    dummies_occupation = pd.get_dummies(user_df[["occupation"]])
    res = pd.concat([dummies_gender, dummies_age], axis=1)
    res.to_pickle("user_df.pkl")
    print(res.head())
