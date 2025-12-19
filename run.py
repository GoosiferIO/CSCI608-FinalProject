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

avg_speeds_by_time_period = speeds.groupby('time_period').agg(
    mean_speed=('speed', 'mean'),
    std_speed=('speed', 'std'),
    count=('speed', 'count')
).reset_index()

# bar chart of average speeds by time period
descriptive_chart = alt.Chart(avg_speeds_by_time_period).mark_bar().encode(
    x=alt.X('time_period:N', 
            sort=['peak', 'offpeak', 'all_day'],
            title='Time Period'),
    y=alt.Y('mean_speed:Q', title='Average Speed (mph)'),
    color='time_period:N'
).properties(
    title='Average Speed by Time Period',
    width=400,
    height=300
)


# make output directory
os.makedirs('outputs', exist_ok=True)

# show chart
descriptive_chart.save('outputs/descriptive_barchart.html')


def main():
    print(speeds.head(10))

if __name__ == "__main__":
    main()