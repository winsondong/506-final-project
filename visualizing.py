import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def load_data(file_path="airbnbListingsData.csv"):
    df = pd.read_csv(file_path)
    
    # Clean the price column
    df["price"] = df["price"].replace(r'[\$,]', '', regex=True).astype(float)

    # Remove extreme price outliers
    df = df[df["price"] < 1000]
    
    return df

def plot_price_histogram(df):
    plt.figure(figsize=(10,6))
    plt.hist(df["price"], bins=50, color='skyblue', edgecolor='black')
    plt.title("Distribution of Airbnb Prices in NYC")
    plt.xlabel("Price ($)")
    plt.ylabel("Number of Listings")
    plt.grid(True)
    plt.show()


def plot_scatter_price_vs_rooms(df):
    plt.figure(figsize=(10, 5))

    # Bedrooms
    plt.subplot(1, 2, 1)
    plt.scatter(df["bedrooms"], df["price"], alpha=0.5)
    plt.title("Price vs Bedrooms")
    plt.xlabel("Bedrooms")
    plt.ylabel("Price ($)")

    # Bathrooms
    plt.subplot(1, 2, 2)
    plt.scatter(df["bathrooms"], df["price"], alpha=0.5, color='orange')
    plt.title("Price vs Bathrooms")
    plt.xlabel("Bathrooms")
    plt.ylabel("Price ($)")

    plt.tight_layout()
    plt.show()



def plot_feature_correlation_heatmap(df):
    numeric_cols = df.select_dtypes(include=["float64", "int64"])
    corr = numeric_cols.corr()

    plt.figure(figsize=(12, 8))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Feature Correlation Heatmap")
    plt.show()


def plot_price_by_borough(df):
    plt.figure(figsize=(10,6))
    sns.boxplot(data=df, x="neighbourhood_group_cleansed", y="price")
    plt.title("Price Distribution by Borough")
    plt.xlabel("Borough")
    plt.ylabel("Price ($)")
    plt.ylim(0, 1000)  # adjust for outliers
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.show()


def plot_top_correlated_features(df, top_n=5):
    numeric_df = df.select_dtypes(include=["float64", "int64"])
    corr = numeric_df.corr()["price"].drop("price").sort_values(key=abs, ascending=False)
    
    plt.figure(figsize=(8,5))
    corr.head(top_n).plot(kind="bar", color="teal", edgecolor="black")
    plt.title("Top Features Correlated with Price")
    plt.ylabel("Correlation Coefficient")
    plt.grid(True)
    plt.show()



if __name__ == "__main__":
    df = load_data()
    plot_price_histogram(df)
    plot_scatter_price_vs_rooms(df)
    plot_feature_correlation_heatmap(df)
    plot_price_by_borough(df)
    plot_top_correlated_features(df)