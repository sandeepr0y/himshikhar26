import seaborn as sns
import matplotlib.pyplot as plt


def heatmap_correlation(df, columns: list, ax):
    corr_matrix = df[columns].corr()
    # plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, ax=ax)
    ax.set_title(f'Correlation Heatmap: {", ".join(columns)}')


def plot_price_distribution(df, ax):
    # plt.figure(figsize=(10, 6))
    ax.hist(df['price'], bins=50, color='skyblue', edgecolor='black')
    ax.set_title('Price Distribution of Used Cars')
    ax.set_xlabel('Price ($)')
    ax.set_ylabel('Frequency')
    ax.grid(axis='y', alpha=0.75)


def plot_all_data(df, ax1, ax2):
    df.plot(kind='scatter', x='price', y='age', color='blue', label='Model Year', ax=ax1)
    ax1.set_title('Price vs Age')
    df.plot(kind='scatter', x='price', y='milage', color='orange', label='Milage', ax=ax2)
    ax2.set_title('Price vs Mileage')

def box_plot(df, *args):
    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    plots = [
        {
            'ax': ax,
            'x': "price",
            'y': "fuel_type",
            'title': '"Price Distribution by Fuel Type"',
            'ylabel': "Fuel Type",
            'xlabel': "Price ($)",
            'palette': "Set2",
            'ticklabel_rotation': 45
        },
        # {
        #     'ax': ax,
        #     'x': "price",
        #     'y': "brand_segment",
        #     'title': "Price Distribution by Segment",
        #     'ylabel': "Brand Segment",
        #     'xlabel': "Price ($)",
        #     'palette': "Pastel1"
        # }
    ]

    for arg in plots:
        sns.boxplot(x=arg['x'], y=arg['y'], data=df, ax=arg['ax'], palette=arg['palette'])
        arg['ax'].set_title(arg['title'])
        arg['ax'].set_xlabel(arg['xlabel'])
        arg['ax'].set_ylabel(arg['ylabel'])
        if 'ticklabel_rotation' in arg:
            arg['ax'].tick_params(axis="x", rotation=arg['ticklabel_rotation'])
    
    plt.tight_layout()
    plt.show()


def plot_visualization(df):
    filtered_df = df[df["price"] <= 150000]

    fig, ax = plt.subplots(2, 2, figsize=(12, 5))

    plot_all_data(filtered_df, ax[0, 0], ax[0, 1])
    plot_price_distribution(filtered_df, ax[1, 0])
    heatmap_correlation(filtered_df, ["age", "milage", "price"], ax[1, 1])

    plt.tight_layout()
    plt.show()

    # box_plot(filtered_df)


def analyze_data(df, data_cleaning_and_formatting):
    # print(df.describe())
    # print(df.head())

    # data cleaning
    df = data_cleaning_and_formatting(df)
    # print(df.info())
    # print(df.head())
    # return

    # visualization
    plot_visualization(df)
    
    # print(f"Data analysis completed. Data loaded from: {data_path}")