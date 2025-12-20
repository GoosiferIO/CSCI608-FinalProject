import pandas as pd
import numpy as np
import altair as alt
import os
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split,  GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def main():
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

    ##################################
    # Predictive
    ##################################

    np.random.seed(42)

    # split data
    df_train, df_test = train_test_split(
        speeds, test_size=0.25, random_state=42
    )

    # define training and testing sets
    xtrain = df_train[['route_length', 'direction']]
    ytrain = df_train['speed']
    xtest = df_test[['route_length', 'direction']]
    ytest = df_test['speed']

    # scaling  preprocessor
    preprocessor = make_column_transformer(
        (StandardScaler(), ['route_length', 'direction'])
    )

    # create pipeline
    pipeline = make_pipeline(
        preprocessor, KNeighborsRegressor()
    )

    # 5-fold cross validation
    param_grid = {
        'kneighborsregressor__n_neighbors': list(range(1, 21))
    }

    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=5,
        n_jobs=-1,
        scoring='neg_root_mean_squared_error'
    )

    grid_search.fit(xtrain, ytrain)
    best_k = grid_search.best_params_['kneighborsregressor__n_neighbors']
    predy_knn = grid_search.predict(xtest)

    print(f"Best k: {best_k}")

    # linear regression for comparison
    linear = LinearRegression()
    linear.fit(xtrain, ytrain)

    # predict
    predy_lr = linear.predict(xtest)

    # evaluate
    # note: apparently my version of sklearn doesn't have the squared parameter so had to find an alternative...
    rmspe_knn = mean_squared_error(ytest, predy_knn) ** (1/2)
    rmspe_lm = mean_squared_error(ytest, predy_lr) ** (1/2)

    mean_speed = ytrain.mean()
    percent_error = (rmspe_knn / mean_speed) * 100

    print(f"Average speed: {mean_speed:.2f} mph")
    print(f"RMSPE as % of mean: {percent_error:.1f}%")
    print(f"KNN RMSPE: {rmspe_knn:.2f} mph")
    print(f"Linear Regression RMSPE: {rmspe_lm:.2f} mph")

    # decide winner (knn vs lm, lower is better)
    if rmspe_knn < rmspe_lm:
        print(f"\nBest model: K-NN (K={best_k})")
        print(f"K-NN is better by {rmspe_lm - rmspe_knn:.2f} mph")
    else:
        print(f"\nBest model: Linear Regression")
        print(f"Linear Regression is better by {rmspe_knn - rmspe_lm:.2f} mph")


if __name__ == "__main__":
    main()