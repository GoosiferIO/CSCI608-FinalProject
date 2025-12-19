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
        "district_name",
        "route_name",
    ]
)

# rename columns
df = df.rename(columns={
	"direction_id":"direction",
	"speed_mph":"speed",
	"Shape_Length":"route_length",
})

# group duplicate route_id rows by taking avg speed
speeds = (
    df.groupby(['org_id', 'agency', 'route_id', 'direction', 'time_period'])
    .agg(
        speed=('speed', 'mean'),
        route_length=('route_length', 'first')
    )
    .reset_index()
)


def main():
    print(speeds.head(10))

if __name__ == "__main__":
    main()