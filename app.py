import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import calendar
import joblib
import os

# Load model components
model = joblib.load('rf_model_light.joblib')
imputer = joblib.load('imputer.joblib')
model_columns = joblib.load('model_columns.joblib')

# Load dataset
df = pd.read_csv("dashboard_data_final.csv")
ordinal_map = {2: "Low", 1: "Medium", 0: "High"}

# Color mapping for prediction styling
color_map = {
    "Low": "red",
    "Medium": "orange",
    "High": "blue"
}

# Initialize Dash app
app = dash.Dash(__name__)
app.title = "Flight Reliability Predictor"

app.layout = html.Div([
    html.H1("✈️ Flight Layover Reliability Predictor", style={'textAlign': 'center'}),

    html.Div([
        html.Label("Month"),
        dcc.Dropdown(
            id='month',
            options=[{'label': calendar.month_name[m], 'value': m} for m in sorted(df['month'].unique())],
            value=1,
            style={'marginBottom': '15px'}
        ),
        html.Label("Airline"),
        dcc.Dropdown(
            id='carrier',
            options=[{'label': c, 'value': c} for c in sorted(df['carrier_name'].unique())],
            value=sorted(df['carrier_name'].unique())[0],
            style={'marginBottom': '15px'}
        ),
        html.Label("Airport"),
        dcc.Dropdown(
            id='airport',
            options=[{'label': a, 'value': a} for a in sorted(df['airport'].unique())],
            value=sorted(df['airport'].unique())[0],
            style={'marginBottom': '25px'}
        ),
    ], style={'width': '50%', 'margin': 'auto'}),

    html.Div(id='prediction-output', style={'textAlign': 'center', 'marginTop': '40px'}),

    html.Hr(),
    html.Div([
        html.H4("How Reliability is Calculated", style={'textAlign': 'center'}),
        html.P("The reliability level is predicted using a composite score derived from multiple flight delay factors from data gathered between 2003-2023. These include:", style={'textAlign': 'center'}),
        html.Ul([
            html.Li("Proportion of delayed flights (delay ratio)"),
            html.Li("Average arrival delay"),
            html.Li("Cancellation and diversion rates"),
            html.Li("Proportions of delay caused by: carriers, weather, NAS, security, and late aircraft")
        ], style={'maxWidth': '700px', 'margin': 'auto'}),
        html.P("Each factor is weighted based on importance and combined to produce a reliability score.", style={'textAlign': 'center'}),
        html.P("Reliability categories are defined as:", style={'textAlign': 'center', 'marginTop': '10px'}),
        html.Ul([
            html.Li("Low: Composite Score ≤ 0.2"),
            html.Li("Medium: 0.2 < Composite Score ≤ 0.4"),
            html.Li("High: Composite Score > 0.4")
        ], style={'maxWidth': '700px', 'margin': 'auto'})
    ], style={
        'padding': '30px',
        'border': '1px solid #ddd',
        'borderRadius': '8px',
        'backgroundColor': '#f9f9f9',
        'marginTop': '40px'
    })
])

@app.callback(
    Output('prediction-output', 'children'),
    Input('month', 'value'),
    Input('carrier', 'value'),
    Input('airport', 'value')
)
def predict(month, carrier, airport):
    match = df[
        (df['month'] == month) &
        (df['carrier_name'] == carrier) &
        (df['airport'] == airport)
    ]

    if not match.empty:
        row = match.iloc[0]
        disclaimer = html.Div("ℹ️ Flight path found.", style={'color': 'gray'})
    else:
        disclaimer = html.Div("⚠️ Flight path not found.", style={'color': 'gray'})
        row = df.mean(numeric_only=True)
        row['month'] = month
        row['carrier_name'] = carrier
        row['airport'] = airport

    # Encode inputs
    input_df = pd.DataFrame([row])
    input_encoded = pd.get_dummies(input_df)
    input_encoded = input_encoded.reindex(columns=model_columns, fill_value=0)
    input_final = pd.DataFrame(imputer.transform(input_encoded), columns=model_columns)

    # Predict
    pred_class = model.predict(input_final)[0]
    pred_proba = model.predict_proba(input_final)[0]
    pred_label = ordinal_map[pred_class]
    prediction_style = {
        'color': color_map[pred_label],
        'fontWeight': 'bold',
        'fontSize': '28px'
    }

    return html.Div([
        html.H2([html.Span(pred_label, style=prediction_style)]),
        html.H4("🔍 Prediction using Random Forest"),
        html.H5("📊 Confidence Levels:"),
        html.P(f"Low: {pred_proba[2]*100:.2f}%"),
        html.P(f"Medium: {pred_proba[1]*100:.2f}%"),
        html.P(f"High: {pred_proba[0]*100:.2f}%"),
        disclaimer
    ])

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port)
