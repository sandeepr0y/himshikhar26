# from unittest import result

import pandas as pd
from datetime import datetime

import joblib
from sklearn.preprocessing import OneHotEncoder, TargetEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error

from utils import get_app_root
# from models.linear_regression import get_pipeline
from models.random_forest import get_pipeline
# from models.xgboost import get_pipeline
from utils.ds import analyze_data


def get_df():
    app_root = get_app_root()
    csv_path = app_root / 'data' / 'used_cars.csv'
    return pd.read_csv(csv_path)


def save_model(model_pipeline, filename='car_price_model.joblib'):
    app_root = get_app_root()
    model_path = app_root / 'models' / filename
    joblib.dump(model_pipeline, model_path)


def get_transformer():
    ohe = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
    te = TargetEncoder(categories='auto', target_type='continuous', cv=5, random_state=42)

    transformer = ColumnTransformer(
        transformers=[
            ('one_hot', ohe, ['fuel_type', 'transmission_clean', 'brand_segment', 'ext_col_clean', 'int_col_clean']),
            ('target_enc', te, ['brand', 'model']),
            ('scaler', StandardScaler(), ['age', 'milage', 'horsepower', 'engine_liters'])
        ],
        remainder='passthrough' # Keep all other numerical columns untouched
    )

    return transformer


def data_cleaning_and_formatting(df):
    current_year = datetime.now().year
    df["milage"] = df["milage"].str.replace(" mi.", "", regex=False).str.replace(",", "", regex=False).astype(float)
    df["price"] = df["price"].str.replace("$", "", regex=False).str.replace(",", "", regex=False).astype(float)
    df = df.dropna(subset=['price'])
    df['age'] = current_year - df['model_year']

    brand_price_df = (
        df.groupby(["brand", "model"])["price"]
        .agg(count="count", average_price="mean")
        .sort_values(by="count", ascending=False)
        .reset_index()
    )
    # print(brand_price_df.describe())
    q33 = brand_price_df['average_price'].quantile(0.33)
    q66 = brand_price_df['average_price'].quantile(0.66)

    # print(f"Budget Cutoff (33rd percentile): ${q33:,.2f}")
    # print(f"Luxury Cutoff (66th percentile): ${q66:,.2f}")

    def categorize_brand(price):
        if price <= q33:
            return 'Budget'
        elif price <= q66:
            return 'Mid-Range'
        else:
            return 'Luxury'
    
    brand_price_df['brand_segment'] = brand_price_df['average_price'].apply(categorize_brand)
    df = df.merge(brand_price_df[['brand', 'model', 'brand_segment']], on=['brand', 'model'], how='left')

    # print(df['accident'].drop_duplicates())
    df['accident'] = df['accident'].map({'No accidents': 0, 'At least 1 accident or damage reported': 1})
    df['accident'] = df['accident'].fillna(0).astype(int)

    df['fuel_type'] = df['fuel_type'].fillna('Electric')
    df['fuel_type'] = df['fuel_type'].replace('not supported', 'Gasoline')
    df['fuel_type'] = df['fuel_type'].replace('–', 'Gasoline')
    # print(df['fuel_type'].drop_duplicates())
    # print(df[df['fuel_type'].isin(['not supported', '–'])])

    df['clean_title'] = df['clean_title'].map({'Yes': 1}).fillna(0).astype(int)

    # print("----- Fuel Type -----", df['fuel_type'].isna().sum())
    # print("----- Engine -----", df['engine'].isna().sum())
    # print("----- Transmission -----", df['transmission'].isna().sum())
    # print("----- Exterior Color -----", df['ext_col'].isna().sum())
    # print("----- Interior Color -----", df['int_col'].isna().sum())
    # print("----- Clean Title -----", df['clean_title'].isna().sum())

    # print(df[df['fuel_type'].isna()]['engine'].value_counts().head(20))

    # print("------> Unique Brands:", df['brand'].nunique())
    # print("------> Unique Models:", df['model'].nunique())
    # print("------> Unique Fuel Types:", df['fuel_type'].nunique())
    # print("------> Unique Engine Types:", df['engine'].nunique())
    # print("------> Unique Transmission Types:", df['transmission'].nunique())
    # print("------> Unique Exterior Colors:", df['ext_col'].nunique())
    # print("------> Unique Interior Colors:", df['int_col'].nunique())

    def simplify_transmission(trans):
        trans = str(trans).lower()
        if 'manual' in trans or 'm/t' in trans:
            return 'Manual'
        elif 'cvt' in trans:
            return 'CVT'
        else:
            return 'Automatic'
    
    df['transmission_clean'] = df['transmission'].apply(simplify_transmission)

    # print(df['engine'].drop_duplicates())
    # Extract Horsepower (numerical value before 'HP')
    df['horsepower'] = df['engine'].str.extract(r"(\d+\.?\d*)\s*HP").astype(float)

    # Extract Liters (numerical value before 'L')
    df['engine_liters'] = df['engine'].str.extract(r"(\d+\.?\d*)\s*(?:L|Liter)").astype(float)

    # Fill any missing values with the median of the dataset
    df['horsepower'] = df['horsepower'].fillna(df['horsepower'].median())
    df['engine_liters'] = df['engine_liters'].fillna(df['engine_liters'].median())

    df['ext_col'] = df['ext_col'].str.lower().str.strip()
    df['int_col'] = df['int_col'].str.lower().str.strip()

    # print(df['int_col'].drop_duplicates())

    def simplify_colour(colour):
        if 'white' in colour:
            return 'white'
        elif 'black' in colour:
            return 'black'
        elif 'silver' in colour:
            return 'silver'
        elif 'gray' in colour or 'grey' in colour:
            return 'gray'
        elif 'blue' in colour or 'blu' in colour:
            return 'blue'
        elif 'red' in colour:
            return 'red'
        elif 'brown' in colour:
            return 'brown'
        elif 'green' in colour:
            return 'green'
        elif 'yellow' in colour:
            return 'yellow'
        elif 'metallic' in colour:
            return 'metallic'
        else:
            return colour
        
    df['ext_col'] = df['ext_col'].replace('–', 'other')
    df['int_col'] = df['int_col'].replace('–', 'other')

    df['ext_col'] = df['ext_col'].apply(simplify_colour)
    df['int_col'] = df['int_col'].apply(simplify_colour)

    # Keep only the top 12 most common exterior colors, group the rest into 'Other'
    top_ext_colors = df[~df['int_col'].isin(['other'])]['ext_col'].value_counts().index[:12]
    top_int_col = df[~df['int_col'].isin(['other'])]['int_col'].value_counts().index[:12]

    df['ext_col_clean'] = df['ext_col'].apply(lambda x: x if x in top_ext_colors else 'other')
    df['int_col_clean'] = df['int_col'].apply(lambda x: x if x in top_int_col else 'other')

    return df


def main():
    df = data_cleaning_and_formatting(get_df())

    # # Filter out the multi-million dollar typo/hypercars to stabilize the linear line
    df = df[df["price"] <= 300000]

    feature_columns = [
        'brand', 'model', 'age', 'milage', 'fuel_type', 'transmission_clean',
        'horsepower', 'engine_liters', 'clean_title', 'brand_segment', 'accident',
        'ext_col_clean', 'int_col_clean'
    ]

    X = df[feature_columns]
    y = df['price']

    # test_size=0.2 reserves 20% for testing. 
    # random_state ensures your splits are identical every time you run the code.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    transformer = get_transformer()
    model_pipeline = get_pipeline(transformer)
    model_pipeline.fit(X_train, y_train)

    # Generate predictions on your unseen test set
    y_pred = model_pipeline.predict(X_test)

    # Calculate Evaluation Metrics
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)

    # print("--- Linear Regression Performance ---")
    print(f"R² Score: {r2:.4f}")
    print(f"Mean Absolute Error (MAE): ${mae:,.2f}")
    print(f"Root Mean Squared Error (RMSE): ${rmse:,.2f}")

    # Comparison table for the first 10 rows of the test set
    comparison_df = pd.DataFrame({
        'Actual Price': y_test.values,
        'Predicted Price': y_pred
    })
    comparison_df['Error Amount'] = comparison_df['Predicted Price'] - comparison_df['Actual Price']

    print("\nSample Predictions Comparison:")
    print(comparison_df.head(10).round(2))

    # save_model(model_pipeline, filename='used_car_price_model.joblib')


if __name__ == "__main__":
    # analyze_data(get_df(), data_cleaning_and_formatting)
    main()