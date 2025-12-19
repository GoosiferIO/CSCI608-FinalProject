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

##################################################
# Explorative
###################################################

# aggregate to get average speed per route
dfexploratory = (
    df.groupby(['org_id', 'route_id', 'direction'])
    .agg(
        speed=('speed', 'mean'),
        route_length=('route_length', 'first'),
        agency=('agency', 'first')
    )
    .reset_index()
)

# scatter plot route length vs avg speed
explore_graph = alt.Chart(dfexploratory).encode(
    x=alt.X('route_length:Q', title='Route Length'),
    y=alt.Y('speed:Q', title='Average Speed (mph)'),
    tooltip=['agency', 'route_id', 'direction', 'route_length', 'speed']
)

scatter = explore_graph.mark_circle(size=60, opacity=0.6, color='steelblue')

# add regression line
regression_line = explore_graph.transform_regression(
    'route_length', 'speed'
).mark_line(color='red', size=2)

# combine
(scatter + regression_line).properties(
    title='Route Length vs. Average Speed',
    width=600,
    height=400
)

# save chart
(scatter + regression_line).save('outputs/explorative_scatter.html')


def main():
    print(speeds.head(10))

if __name__ == "__main__":
    main()