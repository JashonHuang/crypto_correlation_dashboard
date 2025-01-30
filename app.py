import dash
from dash import dcc, html, Input, Output,State,ctx
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import requests
import plotly.express as px
import time 

# intialize the app 
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

MAX_RETRY = 5  #  maxium retry 

# for interval choosing 
INTERVALS  = [1, 5, 15, 30, 60, 240, 1440] # 10080, 21600, less than one day choose


# get tradable assets 
def get_tradable_assets():
    url = "https://api.kraken.com/0/public/AssetPairs"
    response = requests.get(url)
    try_time = 1
    while response.status_code != 200 and try_time <= MAX_RETRY: 
         time.sleep(1) 
         response = requests.get(url)  # Acutally  you can  change proxies , while you have proxies 
         try_time += 1 
    if try_time > MAX_RETRY: 
        return []     
    else:
        data = response.json()
        return [key for key in data['result']]

# get OHLC data 
def get_ohlc_data(pair, interval=1440):  # default 1 day 
    url = f"https://api.kraken.com/0/public/OHLC?pair={pair}&interval={interval}"
    response = requests.get(url)
    try_time = 1
    print(f"collectiong pair of {pair}")
    while response.status_code != 200 and try_time <= MAX_RETRY: 
        print("error while get data!!! %s "%response.status_code)
        time.sleep(1) 
        response = requests.get(url)  # Acutally  you can  change proxies , while you have proxies 
        try_time += 1 
    if try_time > MAX_RETRY: 
        return pd.DataFrame(columns=["time", "close"])
    else:
        data = response.json()
        # print(data)
        ohlc = data['result'][pair]
        df = pd.DataFrame(ohlc, columns=["time", "open", "high", "low", "close", "vwap", "volume", "count"])
        df["close"] = df["close"].astype(float)
        return df[["time", "close"]].sort_values(by='time')
    

# While the serive is running , we can update the asset options in every period, like every hour
def fetch_latest_assets():
    # Simulate fetching from an API
    asset_pairs = get_tradable_assets()
    return asset_pairs
    
# corr df processing methods 
    # Merge all data into a single DataFrame, aligning by "time"
def processing_price_df(filtered_prices):
    price_df = None
    for asset, df in filtered_prices.items():
        if price_df is None:
            price_df = df[["time", "close"]].rename(columns={"close": asset})
        else:
            price_df = pd.merge(price_df, df[["time", "close"]].rename(columns={"close": asset}), on="time", how="outer")

    # ✅ Fill missing values (forward fill, then backward fill)
    price_df = price_df.fillna(method="ffill").fillna(method="bfill")
    # print(price_df.shape)
    # print(price_df.head())

    # If assets have less than 30% of data, drop them
    min_valid_points = int(0.3 * price_df.shape[0])  # Allow at least 30% non-null data
    price_df = price_df.dropna(thresh=min_valid_points, axis=1)
    # print(price_df.head())
    
    return price_df

    # if price_df.shape[1] < 2:
    #     return px.imshow([[0]], text_auto=True, labels=dict(color="Correlation"))

def get_lower_triangular(matrix):
    mask = np.tril(np.ones(matrix.shape), k=-1)  # Mask the lower triangle, keep diagonal
    masked_matrix = pd.DataFrame(
    np.where(mask, matrix, np.nan), # Set lower triangle to NaN
    columns=matrix.columns,
    index=matrix.index
    )
    return masked_matrix 

# Get assets 
asset_pairs = get_tradable_assets()
default_selection = asset_pairs[0:10]


# 创建 Dash 布局
app.layout = dbc.Container([
    html.H1("Cryptocurrency Correlation Matrix", className="text-center mt-4"),
    
    # 用户选择资产,  # Row for Dropdown + Button
    dbc.Row([
        # choose interval 
        dbc.Col([
            html.Label("Interval"),
            dcc.Dropdown(
                id="interval-dropdown",
                options=[{"label": f"{i} min", "value": i} for i in INTERVALS],
                value=1440,  # Default: 1-day interval
            ),
        ], width=2),

        dbc.Col([
            html.Label("Select Asset Pairs"),
            dcc.Dropdown(
                id="asset-dropdown",
                options=[{"label": pair, "value": pair} for pair in asset_pairs],
                multi=True,
                value=default_selection[:10]
            )
        ], width=9),

    

        dbc.Col([
            dbc.Button("Analyse", id="generate-button", color="primary", n_clicks=0, className="d-flex align-items-center justify-content-end")
        ], width=1, className="text-right")
        
    ], className="mb-4"),

    # warning messages, 
     # Warning message area
    html.Div(id="warning-message", style={"color": "red", "margin-top": "10px"}),

    # Store component to track previous valid selection
    dcc.Store(id="stored-selection", data=default_selection),
    dcc.Store(id="stored-interval", data=1440), 


    # Loading Spinner for Correlation Matrix
    dcc.Loading(
        id="loading-spinner",
        type="circle",  # Spinner type ("default", "circle", or "dot")
        children=[
            dbc.Row([
                dbc.Col([dcc.Graph(id="correlation-matrix")], width=12),
            ])
        ]
    ),

    # Interval component to update options every 1 hour (3600 * 1000 ms)
    dcc.Interval(
        id="interval-component",
        interval=3600 * 1000,  # 1 hour in milliseconds
        n_intervals=0  # Start immediately
    )
    ], fluid=True)



# Callback to update dropdown options dynamically
@app.callback(
    Output("asset-dropdown", "options"),
    Input("interval-component", "n_intervals")
)
def update_dropdown_options(n_intervals):
    # Fetch the latest asset options (replace with API request)
    asset_pairs = fetch_latest_assets()
    return asset_pairs


# Callback to enforce asset selection limit
@app.callback(
    [Output("asset-dropdown", "value"),
     Output("warning-message", "children"),
     Output("stored-selection", "data"),
     Output("stored-interval", "data")],
    [Input("asset-dropdown", "value"),
     Input("interval-dropdown", "value")],
    [State("stored-selection", "data"),
     State("stored-interval", "data")]
)
def limit_selection(selected_assets, selected_interval, stored_selection, stored_interval):
    max_items = 10

    if len(selected_assets) > max_items:
        return stored_selection, f"⚠️ You can only select up to {max_items} assets!", stored_selection,selected_interval
    else:
        return selected_assets, "", selected_assets,selected_interval


# Callback to generate correlation matrix when button is clicked
@app.callback(
    Output("correlation-matrix", "figure"),
    Input("generate-button", "n_clicks"),
    [State("stored-selection", "data"),
     State("stored-interval", "data")]
)
def update_correlation_matrix(n_clicks, selected_assets, selected_interval):
    if n_clicks == 0 and not selected_assets:
        return px.imshow([[0]], text_auto=True, labels=dict(color="Correlation"))

    # Filter price data
    prices = {pair: get_ohlc_data(pair, interval=selected_interval) for pair in selected_assets}
    filtered_prices = {pair: prices[pair] for pair in selected_assets if pair in prices}
    if not filtered_prices:
        return px.imshow([[0]], text_auto=True, labels=dict(color="Correlation"))

    # Create correlation matrix

    price_df = processing_price_df(filtered_prices)
    # print(price_df.head())
    
    if price_df.shape[1] < 2:  # no more then 2 symobls 
        return px.imshow([[0]], text_auto=True, labels=dict(color="Correlation"))
    
    price_df = price_df.set_index("time")
    correlation_matrix = price_df.corr(method="pearson")
    # print(correlation_matrix.head())

    # Create an upper triangular mask
    triangular_matrix = get_lower_triangular(correlation_matrix)
    hover_text = triangular_matrix.applymap(lambda x: f"Correlation: {x:.2f}" if not np.isnan(x) else "")

    # Heatmap visualization
    fig = px.imshow(
        triangular_matrix,
        text_auto=True,
        color_continuous_scale="RdBu",
        labels=dict(x="Asset", y="Asset", color="Correlation"),
        x=triangular_matrix.columns,
        y=triangular_matrix.index
    )
    
    fig.update_traces(
        hovertemplate="<b>Asset X:</b> %{x}<br><b>Asset Y:</b> %{y}<br>%{customdata}<extra></extra>",
        customdata=hover_text.values,
        hoverinfo="text"  # Suppresses default hover showing NaN
    )

    fig.update_layout(
        title="Cryptocurrency Correlation Matrix",
        autosize=False,
        xaxis_title="Assets",
        yaxis_title="Assets",
        plot_bgcolor='white'
    )

    return fig

# 运行应用
if __name__ == "__main__":
    print("Starting Dash server...")  # 添加日志
    app.run_server(debug=True)
