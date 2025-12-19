import pandas as pd
import numpy as np
import altair as alt

# cvs read
df = pd.read_csv("data/routespeeds.csv")

##################################################
# Tidy data
###################################################


# drop tables we don't want
df = df.drop(
    columns=[
        "OBJECTID",
        "base64_url",
        "agency",
        "district_name",
        "route_name",
        "org_id",
    ]
)


def main():
    print(df.head())

if __name__ == "__main__":
    main()