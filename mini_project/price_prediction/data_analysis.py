import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime

from utils import get_app_root


def heatmap_correlation(df, columns: list, ax):
    corr_matrix = df[columns].corr()
    # plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, ax=ax)
    ax.set_title(f'Correlation Heatmap: {", ".join(columns)}')


# def plot_filtered_price_distribution(df):
#     print(df["price"].describe())  # Print summary statistics of the original data
#     df_filtered = df[df["price"] <= 100000]
#     print(df_filtered["price"].describe())  # Print summary statistics of the filtered data
#     # plt.figure(figsize=(10, 6))
#     plt.hist(df_filtered['price'], bins=50, color='skyblue', edgecolor='black')
#     plt.title('Price Distribution of Filtered Used Cars')
#     plt.xlabel('Price ($)')
#     plt.ylabel('Frequency')
#     plt.grid(axis='y', alpha=0.75)
#     plt.show()


def plot_price_distribution(df, ax):
    # plt.figure(figsize=(10, 6))
    ax.hist(df['price'], bins=50, color='skyblue', edgecolor='black')
    ax.set_title('Price Distribution of Used Cars')
    ax.set_xlabel('Price ($)')
    ax.set_ylabel('Frequency')
    ax.grid(axis='y', alpha=0.75)


# def plot_specific_brand(df, brand_name, model_name):
#     brand_data = df[(df['brand'] == brand_name) & (df['model'] == model_name)]
    
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

#     brand_data.plot(kind='scatter', x='price', y='age', color='blue', label='Model Year', ax=ax1)
#     ax1.set_title('Price vs Age')

#     brand_data.plot(kind='scatter', x='price', y='milage', color='orange', label='Milage', ax=ax2)
#     ax2.set_title('Price vs Mileage')
    
#     # ax1.legend()
#     # ax2.legend()
#     plt.tight_layout()
#     plt.show()


def plot_all_data(df, ax1, ax2):
    df.plot(kind='scatter', x='price', y='age', color='blue', label='Model Year', ax=ax1)
    ax1.set_title('Price vs Age')
    df.plot(kind='scatter', x='price', y='milage', color='orange', label='Milage', ax=ax2)
    ax2.set_title('Price vs Mileage')


def plot_visualization(df):
    filtered_df = df[df["price"] <= 150000]

    fig, ax = plt.subplots(2, 2, figsize=(12, 5))

    plot_all_data(filtered_df, ax[0, 0], ax[0, 1])
    plot_price_distribution(filtered_df, ax[1, 0])
    heatmap_correlation(filtered_df, ["age", "milage", "price"], ax[1, 1])

    plt.tight_layout()
    plt.show()


def analyze_data():
    app_root = get_app_root()
    csv_path = app_root / 'data' / 'used_cars.csv'
    # print(f"Data file path: {csv_path}")
    
    df = pd.read_csv(csv_path)
    # print(df.describe())
    # print(df.head())

    # data cleaning
    current_year = datetime.now().year
    df["milage"] = df["milage"].str.replace(" mi.", "", regex=False).str.replace(",", "", regex=False).astype(float)
    df["price"] = df["price"].str.replace("$", "", regex=False).str.replace(",", "", regex=False).astype(float)
    df['age'] = current_year - df['model_year']

    # result = (
    #     df.groupby(["brand", "model"])["price"]
    #     .agg(count="count", average_price="mean")
    #     .sort_values(by="count", ascending=False)
    #     .reset_index()
    # )
    # print(result.head())
    # return

    # visualization
    plot_visualization(df)
    # plot_all_data(df)
    # plot_specific_brand(df, "BMW", "M3 Base")
    # plot_specific_brand(df, "Ford", "F-150 XLT")
    # plot_price_distribution(df)
    # plot_filtered_price_distribution(df)
    # heatmap_correlation(df, ["model_year", "milage", "price"])
    
    # print(f"Data analysis completed. Data loaded from: {data_path}")


if __name__ == "__main__":
    analyze_data()