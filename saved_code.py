import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split

st.title("Walmart Retail Inventory Optimization")

# Process state management
if "process_state" not in st.session_state:
    st.session_state.process_state = None

if "model_trained" not in st.session_state:
    st.session_state.model_trained = False

if "forecast_results" not in st.session_state:
    st.session_state.forecast_results = None

# File Upload
uploaded_file = st.sidebar.file_uploader("Upload your Excel file", type=["xlsx"])

if uploaded_file:
    # Load the dataset
    df = pd.read_excel(uploaded_file)

    # Convert transaction_date to datetime format
    df['transaction_date'] = pd.to_datetime(df['transaction_date'])

    # Separate numerical and categorical columns
    numerical_cols = df.select_dtypes(include=['number']).columns
    categorical_cols = df.select_dtypes(include=['object']).columns

    # Aggregate numerical columns by monthly sum
    numerical_agg = df.groupby(pd.Grouper(key='transaction_date', freq='M'))[numerical_cols].sum()

    # Aggregate categorical columns by monthly mode
    def mode_aggregation(series):
        return series.mode()[0] if not series.mode().empty else None

    categorical_agg = df.groupby(pd.Grouper(key='transaction_date', freq='M'))[categorical_cols].agg(mode_aggregation)

    # Merge the aggregated numerical and categorical data
    df_monthly = pd.concat([numerical_agg, categorical_agg], axis=1).reset_index()

    # Encode categorical columns
    for col in categorical_cols:
        if col in df_monthly.columns:
            df_monthly[col] = LabelEncoder().fit_transform(df_monthly[col].astype(str))

    # Define target variable
    y = df_monthly['actual_demand']

    # --- Numerical Features ---
    X_numerical = df_monthly[numerical_cols].drop([
        "transaction_id", "customer_id", "product_id", 'actual_demand'
    ], axis=1, errors='ignore')

    # Train a RandomForest on numerical features
    rf_numerical = RandomForestRegressor()
    rf_numerical.fit(X_numerical, y)

    # Get feature importances for numerical features
    numerical_importances = rf_numerical.feature_importances_
    numerical_feature_names = X_numerical.columns

    # --- Categorical Features ---
    X_categorical = df_monthly[categorical_cols].drop(['actual_demand'], axis=1, errors='ignore')

    # Train a RandomForest on categorical features
    rf_categorical = RandomForestRegressor()
    rf_categorical.fit(X_categorical, y)

    # Get feature importances for categorical features
    categorical_importances = rf_categorical.feature_importances_
    categorical_feature_names = X_categorical.columns

    # Save process state
    st.session_state.process_state = {
        "df_monthly": df_monthly,
        "numerical_importances": numerical_importances,
        "numerical_feature_names": numerical_feature_names,
        "categorical_importances": categorical_importances,
        "categorical_feature_names": categorical_feature_names
    }

    st.success("Process state saved. You can now proceed to the next session.")
else:
    st.warning("Please upload an Excel file to proceed.")

# --- Separate Session for Visualization ---
if st.session_state.process_state:
    state = st.session_state.process_state

    df_monthly = state["df_monthly"]
    numerical_importances = state["numerical_importances"]
    numerical_feature_names = state["numerical_feature_names"]
    categorical_importances = state["categorical_importances"]
    categorical_feature_names = state["categorical_feature_names"]

    st.subheader("Numerical Feature Importance")
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    sns.barplot(x=numerical_importances, y=numerical_feature_names, ax=ax1)
    ax1.set_title("Numerical Feature Importance")
    st.pyplot(fig1)

    st.subheader("Categorical Feature Importance")
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    sns.barplot(x=categorical_importances, y=categorical_feature_names, ax=ax2)
    ax2.set_title("Categorical Feature Importance")
    st.pyplot(fig2)

    # Keep only the relevant columns and retain 'transaction_date' as the index
    final_columns = ['store_location', 'weekday', 'weather_conditions', 'customer_age', 'customer_income',
                     'supplier_lead_time', 'reorder_quantity', 'unit_price', 'quantity_sold', 'actual_demand']

    df_final = df_monthly[final_columns]

    # Set 'transaction_date' as the index
    df_final['transaction_date'] = df_monthly['transaction_date']
    df_final.set_index('transaction_date', inplace=True)

    st.subheader("Final Processed Data")
    st.write(df_final)

    # Option to download the final dataframe as CSV
    csv = df_final.to_csv().encode('utf-8')
    st.download_button(label="Download Processed Data as CSV", data=csv, file_name='processed_data.csv', mime='text/csv')

    # Plot Actual Demand Over Time
    st.subheader("Actual Demand Over Time")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_final.index, 
        y=df_final['actual_demand'], 
        mode='lines', 
        name='Actual Demand',
        line=dict(color='blue')
    ))
    fig.update_layout(
        title='Actual Demand Over Time',
        xaxis_title='Date',
        yaxis_title='Actual Demand',
        template='plotly_white'
    )
    st.plotly_chart(fig)

    # --- Start New Session for LSTM Model ---
    st.subheader("LSTM Model Training Session")

    index = df_final.index
    scaler = MinMaxScaler()
    scaled_values = scaler.fit_transform(df_final.drop(['store_location', 'weekday', 'weather_conditions'], axis=1))
    scaled_df = pd.DataFrame(scaled_values, columns=['customer_age', 'customer_income', 'supplier_lead_time',
                                                      'reorder_quantity', 'unit_price', 'quantity_sold', 'actual_demand'])

    scaled_df[['store_location', 'weekday', 'weather_conditions']] = df_final[['store_location', 'weekday', 'weather_conditions']].values
    scaled_df.index = index

    x = scaled_df.drop(["actual_demand"], axis=1)
    y = scaled_df["actual_demand"]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # Convert DataFrame to NumPy arrays for LSTM model
    x_train = x_train.values
    x_test = x_test.values

    # Reshape data for LSTM: (samples, time steps, features)
    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
    x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

    # Build the LSTM model
    if not st.session_state.model_trained:
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=False, input_shape=(x_train.shape[1], 1)))
        model.add(Dense(units=1))  # Output layer for predicting demand

        # Compile the model
        model.compile(optimizer='adam', loss='mean_squared_error')

        # Train the model
        history = model.fit(x_train, y_train, epochs=51, batch_size=1, validation_data=(x_test, y_test))

        st.session_state.model = model
        st.session_state.model_trained = True
        st.success("LSTM model training completed.")

    else:
        model = st.session_state.model
        st.info("Model already trained.")

    # Define the demand scaler
    demand_scaler = MinMaxScaler()
    demand_scaler.fit(df_final[['actual_demand']])

    st.write(f"Scaler Min: {demand_scaler.data_min_[0]}, Max: {demand_scaler.data_max_[0]}")

    # Take the last 9 months of test data to forecast the next months
    last_sequence = x_test[-1].reshape(1, 9, 1)

    # Initialize the list to store the predicted values
    future_predictions = []

    # Forecast for the next months
    months = st.number_input("How many months to forecast:", min_value=1, max_value=12, value=6, step=1)
    for _ in range(months):
        # Predict the next month's demand
        next_pred_scaled = model.predict(last_sequence, verbose=0)
        future_predictions.append(next_pred_scaled[0][0])
        last_sequence = np.roll(last_sequence, shift=-1, axis=1)
        last_sequence[0, -1, 0] = next_pred_scaled[0][0]

    # Rescale the forecasted demand values back to the original scale
    future_predictions = demand_scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

    st.subheader("Forecasted Demand")
    st.write(future_predictions)

    # Plot the forecasted demand
    forecast_dates = pd.date_range(start=df_final.index[-1], periods=months + 1, freq='M')[1:]
    fig_forecast = go.Figure()
    fig_forecast.add_trace(go.Scatter(
        x=forecast_dates, 
        y=future_predictions.flatten(), 
        mode='lines+markers', 
        name='Forecasted Demand',
        line=dict(color='orange')
    ))
    fig_forecast.update_layout(
        title='Forecasted Demand for Upcoming Months',
        xaxis_title='Date',
        yaxis_title='Demand',
        template='plotly_white'
    )
    st.plotly_chart(fig_forecast)
