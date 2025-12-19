import pandas as pd
import numpy as np
import altair as alt
import os

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

##################################################
# Descriptive 
###################################################

descriptive_chart = alt.Chart(speeds).mark_boxplot().encode(
    x=alt.X('time_period:N', 
            sort=['peak', 'offpeak', 'all_day'],
            title='Time Period'),
    y=alt.Y('speed:Q', title='Speed (mph)'),
    color='time_period:N'
).properties(
    title='Speed Distribution by Time Period for All Routes',
    width=400,
    height=300
)


def main():
    print(speeds.head(10))

if __name__ == "__main__":
    main()