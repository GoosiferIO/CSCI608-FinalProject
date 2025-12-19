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

# rename columns
df = df.rename(columns={
	"direction_id":"direction",
	"speed_mph":"speed",
	"Shape_Length":"route_length",
})

# group duplicate route_id rows by taking avg speed
avg_speed = (
    df.groupby(["route_id", "time_period"])
    .agg(mean_speed=("speed", "mean"))
    .reset_index()
)


def main():
    print(avg_speed.head())

if __name__ == "__main__":
    main()